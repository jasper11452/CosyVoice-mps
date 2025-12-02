# Copyright (c) 2024 CosyVoice MPS
# Placeholder discriminator module for config compatibility (not used in inference)

import torch
import torch.nn as nn


class MultipleDiscriminator(nn.Module):
    """
    Placeholder discriminator for config compatibility.
    Not used during inference - only needed for training.
    """
    
    def __init__(self, mpd=None, mrd=None, **kwargs):
        super().__init__()
        # These are ignored - discriminator is only for training
        pass
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Discriminator is not used in inference mode")


class MultiResSpecDiscriminator(nn.Module):
    """
    Placeholder multi-resolution spectrogram discriminator.
    Not used during inference.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        pass
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Discriminator is not used in inference mode")
