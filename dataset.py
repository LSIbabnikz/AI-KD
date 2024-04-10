
import os
import pickle

from torchvision.transforms import Compose
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

from utils import *
from alignment import align_image

class WrapperDataset(Dataset):

    def __init__(self,
                 image_location: str,
                 quality_location: str,
                 landms_location: str,
                 trans : Compose,
                 misalign : bool,
                 val_percent : float
        ):
        """Helper class used to construct the training and validation dataset.

        Args:
            image_location (str): Location of training set images.
            quality_location (str): Location of quality labels.
            landms_location (str): Location of landmark data of training images.
            trans (Compose): Transformation used prior to passing images.
            misalign (bool): If True landmarks for the student model are sampled around the baselines.
            val_percent (float): Percentage of images reserved for the validation set.
        """

        super().__init__()

        # Read landmark data
        with open(landms_location, "rb") as pkl_in:
            lanmds = {k: v[1] for k, v in pickle.load(pkl_in).items()}
        # Read quality scores
        with open(quality_location, "rb") as pkl_in:
            qualities = pickle.load(pkl_in)

        # Construct data for training samples 
        all_items = []
        for (dir, subdirs, files) in os.walk(image_location):
            for file in files:
                path = os.path.join(dir, file)
                name = path.split("vggface2/")[-1]
                if name in lanmds and name in qualities:
                    all_items.append((name, path, qualities[name], lanmds[name]))

        # Sort samples by quality scores 
        all_items.sort(key=lambda x: x[2])

        # Divide into validation and train splits
        val_items = all_items[::int(len(all_items)/(len(all_items) * val_percent))]
        train_items = list(set(all_items) - set(val_items))

        # Construct training and valiation datasets
        self.train_set = PartialDataset(train_items,
                                        trans,
                                        misalign)
        self.val_set = PartialDataset(val_items,
                                      trans,
                                      misalign, 
                                      val=1)

    def __call__(self):
        return self.train_set, self.val_set


class PartialDataset(Dataset):

    def __init__(self, 
                 item_list : list, 
                 trans : Compose, 
                 misalign: bool,
                 val: bool = 0):
        """ Dataset class returning properly and misaligned images as well as the baseline quality scores

        Args:
            item_list (list): List of all used items.
            trans (Compose): Transformation used by the student/teacher model.
            misalign (bool): If True landmarks for the student model are sampled around the baselines.
            val (bool): If True this returns validation images.
        """
        super().__init__()

        self.items = item_list
        self.trans = trans
        self.misalign = misalign
        self.val = val

        self.alignment_func = align_image

    def __getitem__(self, x):

        # Load item by index and open image 
        name, loc, quality, landms = self.items[x]
        student_img = Image.open(loc).convert("RGB")
        # Properly aligned image
        proper_align = Image.fromarray(self.alignment_func(np.array(student_img), landms))
        # Sample new landmark for misaligned image
        if self.misalign and not self.val:
            landms += (np.random.random((5,2)) * 2. - 1.) * 3.
        # Misaligned image
        student_img = Image.fromarray(self.alignment_func(np.array(student_img), landms))

        return self.trans(proper_align), self.trans(student_img), quality
    
    def __len__(self):
        return len(self.items)


class InferenceDatasetWrapper():
    
    def __init__(self, 
                 image_loc, 
                 trans) -> None:
        """ Helper class that loads all images from a given directory.

        Args:
            image_loc (str): The location of the directory containing the desired images.
            trans (Compose): Transformations used on loaded images.
        """
        self.image_loc = image_loc
        self.trans = trans

        self.items = []
        for (dir, subdirs, files) in os.walk(self.image_loc):
            self.items.extend([os.path.join(dir, file) for file in files if isimagefile(file)])

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, x):
        path = self.items[x]
        return (path, self.trans(Image.open(path).convert("RGB")))

