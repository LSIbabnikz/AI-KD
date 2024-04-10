

import os
import argparse
from typing import Callable
from functools import partial
from contextlib import nullcontext

import wandb
from tqdm import tqdm

import torch

from utils import *
from dataset import *


def train_and_validate(
        args : Arguments,
        epoch : int,
        student_model : torch.nn.Module,
        teacher_model : torch.nn.Module,
        train_dataloader : torch.utils.data.DataLoader,
        val_dataloader : torch.utils.data.DataLoader,
        optimizer : torch.optim.Optimizer,
        loss_fn_quality : torch.nn.Module,
        loss_fn_consistency : torch.nn.Module,
        grad_scaler : torch.cuda.amp.GradScaler,
        wandb_logger,
        ema_func : Callable,
        ema_scheduler : torch.optim.swa_utils.SWALR,
        ema_model: torch.nn.Module,
        pretrain_end : int,
        warmup_end : int,
        best_val_loss : float,
        total_steps : int,
        val_every : int,
        scheduler_pretrain : torch.optim.lr_scheduler.LRScheduler
        ) -> None:
    """ Main training script for AI-KD.

    Args:
        args (Arguments): Arguments from the given training script.
        epoch (int): Current epoch.
        student_model (torch.nn.Module): Student model trained using AI-KD.
        teacher_model (torch.nn.Module): Teacher model used for feature alignment.
        train_dataloader (torch.utils.data.DataLoader): Train dataloader.
        val_dataloader (torch.utils.data.DataLoader): Validation dataloader.
        optimizer (torch.optim.Optimizer): Used optimizer.
        loss_fn_quality (torch.nn.Module): Quality loss function.
        loss_fn_consistency (torch.nn.Module): Feature alignment loss function.
        grad_scaler (torch.cuda.amp.GradScaler): Gradient Scaler.
        wandb_logger (_type_): Wandb logger.
        ema_func (function): EMA model update function.
        ema_scheduler (torch.optim.swa_utils.SWALR): EMA model scheduler.
        ema_model (torch.nn.Module): EMA model.
        pretrain_end (int): End step of pretraining (for schedulers).
        warmup_end (int): End step of LR scheduler warmup.
        best_val_loss (float): Current best validation loss.
        total_steps (int): Current number of total steps.
        val_every (int): Number of steps after which validation is done.
        scheduler_pretrain (torch.optim.lr_scheduler.LRScheduler): LR scheduler.
    """

    teacher_model.eval()
    student_model.train()

    for (teacher_image_batch, student_image_batch, quality_batch) in (pbar := tqdm(train_dataloader, 
                                           desc=f" Training Epoch ({epoch}/{args.base.epochs}), Loss: NaN ", 
                                           disable=not args.base.verbose)):

        teacher_image_batch = teacher_image_batch.to(args.base.device)
        student_image_batch = student_image_batch.to(args.base.device)
        quality_batch = quality_batch.to(args.base.device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast() if grad_scaler else nullcontext():
            mis_feature_batch, out_quality_batch = student_model(student_image_batch)
            out_quality_batch = out_quality_batch.squeeze()

            # Get quality factor from relative position of input sample w.r.t the precomputed class center
            with torch.no_grad():
                pro_feature_batch = teacher_model(teacher_image_batch)

            # Get both loss terms
            loss_quality = loss_fn_quality(out_quality_batch, quality_batch)
            loss_consistency = loss_fn_consistency(mis_feature_batch, 
                                                   pro_feature_batch, 
                                                   target=torch.tensor([1]).cuda())
        
            loss = loss_consistency + loss_quality

        # Scale/clip gradients and backprop
        if grad_scaler:
            grad_scaler.scale(loss).backward() 
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5)
            grad_scaler.step(optimizer) 
            grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5)
            optimizer.step()

        # Add number of steps
        total_steps += 1
        if total_steps < pretrain_end: # Possibly forward the LR scheduler
            scheduler_pretrain.step()

        pbar.set_description(f" Training Epoch ({epoch}/{args.base.epochs}), Loss: {loss.item():.4f} ")

        if args.wandb.use:
            wandb_logger.log({"Feature Loss": loss_consistency.item(),
                              "Quality Loss": loss_quality.item()})

        # If stepcount equals validation step proceed with the validation
        if total_steps % val_every == 0 and total_steps > pretrain_end:

            best_val_loss = \
                validate(
                    args,
                    student_model,
                    val_dataloader,
                    loss_fn_quality,
                    grad_scaler,
                    wandb_logger,
                    ema_func,
                    ema_scheduler,
                    ema_model,
                    best_val_loss,
                )

    return best_val_loss, total_steps


@torch.no_grad()
def validate(args : Arguments,
             student_model : torch.nn.Module,
             val_dataloader : torch.utils.data.DataLoader,
             loss_fn_quality : torch.nn.Module,
             grad_scaler :  torch.cuda.amp.GradScaler,
             wandb_logger,
             ema_func : Callable,
             ema_scheduler : torch.optim.swa_utils.SWALR,
             ema_model : torch.optim.swa_utils.AveragedModel,
             best_val_loss : float,
             ) -> float:
    """  Validation function used to validate AI-KD models.

    Args:
        args (Arguments): Arguments from the given training script.
        student_model (torch.nn.Module): Student model trained using AI-KD.
        teacher_model (torch.nn.Module): Teacher model used for feature alignment.
        val_dataloader (torch.utils.data.DataLoader): Validation dataloader.
        loss_fn_quality (torch.nn.Module): Quality loss function.
        grad_scaler (torch.cuda.amp.GradScaler): Gradient Scaler.
        wandb_logger (_type_): Wand logger.
        ema_func (function): EMA model update function.
        ema_scheduler (torch.optim.swa_utils.SWALR): EMA model scheduler.
        ema_model (torch.optim.swa_utils.AveragedModel): EMA model.
        best_val_loss (float): Current best validation loss.

    Returns:
        float: Best validation loss.
    """

     # Create the EMA model if first time evaluating
    if ema_model is None:
        ema_model = torch.optim.swa_utils.AveragedModel(student_model, avg_fn=ema_func).to(args.base.device)
    ema_model.update_parameters(student_model)

    ema_scheduler.step()

    student_model.eval()
    ema_model.eval()

    # Validate over the quality scores
    current_val_loss = 0.
    for (teacher_image_batch, student_image_batch, quality_batch) in tqdm(val_dataloader, 
                                                                          total=len(val_dataloader), 
                                                                          disable=not args.base.verbose):

        teacher_image_batch = teacher_image_batch.to(args.base.device)
        student_image_batch = student_image_batch.to(args.base.device)
        quality_batch = quality_batch.to(args.base.device)

        with torch.cuda.amp.autocast() if grad_scaler else nullcontext(), torch.no_grad():

            _, out_quality_batch = student_model(student_image_batch)
            out_quality_batch = out_quality_batch.squeeze()

            loss = loss_fn_quality(out_quality_batch, quality_batch)
            current_val_loss += loss.item()

    current_val_loss /= len(val_dataloader)

    if args.wandb.use:
        wandb_logger.log({"Validation Loss": current_val_loss})

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        torch.save(ema_model.module.state_dict(), os.path.join(args.base.save_path, f"aikd-best.pth"))

    student_model.train()
    ema_model.train()

    return best_val_loss


def main(args):

    # Seed all libraries to ensure consistency between runs.
    seed_all(args.base.seed)

    # Check if save path valid
    assert os.path.exists(args.base.save_path), f"Path {args.base.save_path} does not exist"

    # Load the training FR model and construct the transformation
    student_model, teacher_model, trans = construct_full_model(args.model.config)
    student_model.to(args.base.device)
    teacher_model.to(args.base.device)

    # Construct validation and training dataloaders
    train_dataset, val_dataset = WrapperDataset(**args_to_dict(args.dataset, {}), trans=trans)()
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   **args_to_dict(args.dataloader.train.params, {}))
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 **args_to_dict(args.dataloader.val.params, {}))

    # Create optimizer from config
    optimizer = construct_optimizer(args.optimizer, student_model)

    # Create EMA model
    ema_func = lambda averaged_model_parameter, model_parameter, num_averaged: args.ema.ema_alpha * averaged_model_parameter + (1. - args.ema.ema_alpha) * model_parameter
    ema_scheduler = load_module(args.ema, call=False)
    ema_scheduler = ema_scheduler(optimizer, **args_to_dict(args.ema.params, {})) 
    ema_model = None

    # Load desired loss function
    loss_fn_quality = load_module(args.loss.quality)
    loss_fn_consistency = load_module(args.loss.consistency)

    # Construct WANDB logger
    wandb_logger = None
    if args.wandb.use:
        wandb_logger = wandb.init(project=args.wandb.project, config={"args": args_to_dict(args, {})})

    # Training variables
    pretrain_end = int(len(train_dataloader) * args.base.pretrain)
    warmup_end = int(pretrain_end * args.base.warmup)
    best_val_loss = float("inf")
    total_steps = 0
    val_every = int(args.base.validate_every * len(train_dataloader))

    # Define the lr schedulers
    pretrain_lr_lambda = partial(base_lambdalr, warmup_end=warmup_end)
    scheduler_pretrain = load_module(args.scheduler, call=False)
    scheduler_pretrain = scheduler_pretrain(optimizer, [pretrain_lr_lambda, pretrain_lr_lambda])

    # Use AMP if specified in config
    grad_scaler = None
    if args.base.amp:
        grad_scaler = load_module(args.grad_scaler, call=False)
        grad_scaler = grad_scaler(growth_interval=pretrain_end,
                                  **args_to_dict(args.grad_scaler.params, {}))

    # Train loop
    for epoch in range(args.base.epochs):

        best_val_loss, total_steps = \
                train_and_validate(
                    args, 
                    epoch, 
                    student_model, 
                    teacher_model, 
                    train_dataloader,
                    val_dataloader, 
                    optimizer, 
                    loss_fn_quality, 
                    loss_fn_consistency, 
                    grad_scaler, 
                    wandb_logger,
                    ema_func,
                    ema_scheduler,
                    ema_model,
                    pretrain_end,
                    warmup_end,
                    best_val_loss,
                    total_steps,
                    val_every,
                    scheduler_pretrain
                    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", 
        "--config",
        type=str,
        help=' Location of the AI-KD training configuration. '
    )
    args = parser.parse_args()
    arguments = parse_config_file(args.config)

    main(arguments)
