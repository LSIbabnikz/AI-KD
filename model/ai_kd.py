
import torch

class WrapperModel(torch.nn.Module):

    def __init__(self, 
                 feature_extractor : torch.nn.Module, 
                 quality_head : torch.nn.Module
                 ) -> None:
        """Wrapper model for AI-KD consisting of a feature extraction backbone and a simple quality head MLP.

        Args:
            feature_extractor (torch.nn.Module): Feature extraction backbone.
            quality_head (torch.nn.Module): Quality regression head.
        """
        super().__init__()


        self.feature_extractor = feature_extractor
        self.quality_head = quality_head

    def forward(self, x):
        feature = self.feature_extractor(x)
        quality_bins = self.quality_head(feature)

        return feature, quality_bins