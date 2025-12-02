# Copyright (c) 2024 CosyVoice MPS
# Minimal HiFiGan wrapper for inference only (no discriminator/training code)

import torch
import torch.nn as nn


class HiFiGan(nn.Module):
    """
    Minimal HiFiGan wrapper for inference.
    
    The original implementation includes discriminator and training logic,
    but for MPS inference we only need the generator.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module = None,  # ignored for inference
        mel_spec_transform: list = None,  # ignored for inference
    ):
        super().__init__()
        self.generator = generator
        # discriminator and mel_spec_transform are only used for training
        # we keep the parameters for config compatibility but don't use them
    
    def forward(self, *args, **kwargs):
        """Forward pass through generator."""
        return self.generator(*args, **kwargs)
    
    def inference(self, *args, **kwargs):
        """Inference pass through generator."""
        return self.generator.inference(*args, **kwargs)
    
    def remove_weight_norm(self):
        """Remove weight normalization from generator."""
        if hasattr(self.generator, 'remove_weight_norm'):
            self.generator.remove_weight_norm()
