#!/usr/bin/env python
"""
MPS compatibility wrapper for diffusers Attention.

This module provides a wrapper that moves attention computation to CPU
when running on MPS, as the diffusers Attention module can produce NaN
values on Apple Silicon.
"""
import torch
import torch.nn as nn
from functools import wraps


def mps_cpu_fallback(func):
    """Decorator that moves computation to CPU for MPS tensors if needed."""
    @wraps(func)
    def wrapper(self, hidden_states, *args, **kwargs):
        # Check if we're on MPS and if so, move to CPU
        if hidden_states.device.type == 'mps':
            # Move inputs to CPU
            hidden_states_cpu = hidden_states.cpu()
            
            # Handle attention_mask if present
            attention_mask = kwargs.get('attention_mask', None)
            if attention_mask is not None:
                kwargs['attention_mask'] = attention_mask.cpu()
            
            # Handle encoder_hidden_states if present
            encoder_hidden_states = kwargs.get('encoder_hidden_states', None)
            if encoder_hidden_states is not None:
                kwargs['encoder_hidden_states'] = encoder_hidden_states.cpu()
            
            # Run on CPU
            output = func(self, hidden_states_cpu, *args, **kwargs)
            
            # Move back to MPS
            return output.to(hidden_states.device)
        else:
            return func(self, hidden_states, *args, **kwargs)
    
    return wrapper


def patch_attention_for_mps():
    """
    Patch the diffusers Attention class to use CPU fallback on MPS.
    Call this before creating any model that uses diffusers Attention.
    """
    try:
        from diffusers.models.attention_processor import Attention
        
        # Store original forward
        _original_forward = Attention.forward
        
        @mps_cpu_fallback
        def forward_with_mps_fallback(self, hidden_states, *args, **kwargs):
            return _original_forward(self, hidden_states, *args, **kwargs)
        
        Attention.forward = forward_with_mps_fallback
        print("✓ Patched diffusers Attention for MPS CPU fallback")
        
    except ImportError:
        print("⚠ diffusers not installed, skipping Attention patch")


def patch_transformer_block_for_mps():
    """
    Patch the BasicTransformerBlock to use CPU for attention on MPS.
    """
    try:
        from matcha.models.components.transformer import BasicTransformerBlock
        
        _original_forward = BasicTransformerBlock.forward
        
        def forward_with_mps_safety(self, hidden_states, attention_mask=None, **kwargs):
            original_device = hidden_states.device
            
            if original_device.type == 'mps':
                # Move everything to CPU
                hidden_states = hidden_states.cpu()
                if attention_mask is not None:
                    attention_mask = attention_mask.cpu()
                
                # Process kwargs
                cpu_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        cpu_kwargs[k] = v.cpu()
                    else:
                        cpu_kwargs[k] = v
                
                # Run forward on CPU
                output = _original_forward(self, hidden_states, attention_mask, **cpu_kwargs)
                
                # Move back to MPS
                return output.to(original_device)
            else:
                return _original_forward(self, hidden_states, attention_mask, **kwargs)
        
        BasicTransformerBlock.forward = forward_with_mps_safety
        print("✓ Patched BasicTransformerBlock for MPS CPU fallback")
        
    except ImportError:
        print("⚠ matcha not installed, skipping BasicTransformerBlock patch")


def apply_all_mps_patches():
    """Apply all MPS compatibility patches."""
    print("Applying MPS compatibility patches...")
    patch_attention_for_mps()
    patch_transformer_block_for_mps()
    print("Done.")


if __name__ == "__main__":
    # Test the patches
    import sys
    sys.path.insert(0, '/Users/jasper/CosyVoice')
    sys.path.insert(0, '/Users/jasper/CosyVoice/third_party/Matcha-TTS')
    
    apply_all_mps_patches()
    
    # Now test with a simple example
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nTesting on device: {device}")
    
    from matcha.models.components.transformer import BasicTransformerBlock
    
    block = BasicTransformerBlock(
        dim=256,
        num_attention_heads=4,
        attention_head_dim=64,
        dropout=0.0,
        activation_fn="snake",
    ).to(device).eval()
    
    # Test data
    hidden = torch.randn(2, 100, 256, device=device)
    
    with torch.no_grad():
        output = block(hidden)
        print(f"Output shape: {output.shape}")
        print(f"Has NaN: {torch.isnan(output).any().item()}")
