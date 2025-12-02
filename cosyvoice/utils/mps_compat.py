# Copyright (c) 2024 MPS Compatibility Layer for CosyVoice
#
# This module provides compatibility functions for running CosyVoice on Apple Silicon (MPS)
# by handling operations that are not fully supported on the MPS backend.

import torch
import functools
from typing import Callable, Any

# List of operations known to have issues on MPS
# - torch.istft: aten::unfold_backward not supported
# - torch.stft: generally works but may have edge cases  
# - torch.multinomial: can have issues with certain distributions
# - Complex number operations: partial support


def is_mps_device(tensor_or_device) -> bool:
    """Check if the tensor or device is MPS."""
    if isinstance(tensor_or_device, torch.Tensor):
        return tensor_or_device.device.type == 'mps'
    elif isinstance(tensor_or_device, torch.device):
        return tensor_or_device.type == 'mps'
    elif isinstance(tensor_or_device, str):
        return 'mps' in tensor_or_device
    return False


def mps_safe_istft(real: torch.Tensor, img: torch.Tensor, n_fft: int, hop_length: int, 
                   win_length: int, window: torch.Tensor) -> torch.Tensor:
    """
    MPS-safe inverse STFT operation.
    
    MPS doesn't fully support istft (aten::unfold_backward), so we run it on CPU
    to avoid NaN values from mixed device computation.
    """
    original_device = real.device
    
    if is_mps_device(original_device):
        real = real.cpu()
        img = img.cpu()
        window = window.cpu()
    else:
        window = window.to(real.device)
    
    inverse_transform = torch.istft(
        torch.complex(real, img), 
        n_fft, 
        hop_length,
        win_length, 
        window=window
    )
    
    if is_mps_device(original_device):
        inverse_transform = inverse_transform.to(original_device)
    
    return inverse_transform


def mps_safe_stft(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int,
                  window: torch.Tensor, return_complex: bool = True):
    """
    MPS-safe STFT operation.
    
    While STFT generally works on MPS, this provides a fallback if issues occur.
    """
    original_device = x.device
    
    # STFT generally works on MPS, but keep this for consistency
    try:
        spec = torch.stft(
            x, n_fft, hop_length, win_length, 
            window=window.to(x.device),
            return_complex=return_complex
        )
        return spec
    except RuntimeError as e:
        if is_mps_device(original_device):
            # Fallback to CPU
            x_cpu = x.cpu()
            window_cpu = window.cpu()
            spec = torch.stft(
                x_cpu, n_fft, hop_length, win_length,
                window=window_cpu,
                return_complex=return_complex
            )
            return spec.to(original_device)
        raise e


def mps_safe_multinomial(probs: torch.Tensor, num_samples: int, replacement: bool = False) -> torch.Tensor:
    """
    MPS-safe multinomial sampling.
    
    torch.multinomial can have issues on MPS with certain probability distributions.
    """
    original_device = probs.device
    
    if is_mps_device(original_device):
        probs_cpu = probs.cpu()
        result = torch.multinomial(probs_cpu, num_samples, replacement=replacement)
        return result.to(original_device)
    
    return torch.multinomial(probs, num_samples, replacement=replacement)


def mps_safe_topk(input: torch.Tensor, k: int, dim: int = -1, largest: bool = True, 
                  sorted: bool = True):
    """
    MPS-safe topk operation.
    
    topk generally works on MPS but this provides consistency.
    """
    return torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)


def run_on_cpu_if_mps(func: Callable) -> Callable:
    """
    Decorator that runs a function on CPU if the first tensor argument is on MPS.
    Useful for wrapping operations that don't work well on MPS.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Find the first tensor argument
        first_tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                first_tensor = arg
                break
        if first_tensor is None:
            for v in kwargs.values():
                if isinstance(v, torch.Tensor):
                    first_tensor = v
                    break
        
        if first_tensor is not None and is_mps_device(first_tensor.device):
            original_device = first_tensor.device
            
            # Move all tensor args to CPU
            cpu_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    cpu_args.append(arg.cpu())
                else:
                    cpu_args.append(arg)
            
            cpu_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    cpu_kwargs[k] = v.cpu()
                else:
                    cpu_kwargs[k] = v
            
            result = func(*cpu_args, **cpu_kwargs)
            
            # Move result back to original device
            if isinstance(result, torch.Tensor):
                return result.to(original_device)
            elif isinstance(result, tuple):
                return tuple(r.to(original_device) if isinstance(r, torch.Tensor) else r for r in result)
            return result
        
        return func(*args, **kwargs)
    
    return wrapper


class MPSCompatibleModule(torch.nn.Module):
    """
    Base class for modules that need MPS compatibility.
    Provides helper methods for device-aware operations.
    """
    
    def _get_device(self) -> torch.device:
        """Get the device of the first parameter."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')
    
    def _is_mps(self) -> bool:
        """Check if module is on MPS device."""
        return is_mps_device(self._get_device())
    
    def _safe_istft(self, real: torch.Tensor, img: torch.Tensor, n_fft: int, 
                    hop_length: int, win_length: int, window: torch.Tensor) -> torch.Tensor:
        """Perform MPS-safe ISTFT."""
        return mps_safe_istft(real, img, n_fft, hop_length, win_length, window)


# Monkey-patch torch functions for global MPS compatibility (optional, use with caution)
_original_istft = torch.istft

def patched_istft(input, n_fft, hop_length=None, win_length=None, window=None, 
                  center=True, normalized=False, onesided=None, length=None, 
                  return_complex=False):
    """Patched istft that handles MPS automatically."""
    if is_mps_device(input.device):
        input_cpu = input.cpu()
        window_cpu = window.cpu() if window is not None else None
        result = _original_istft(
            input_cpu, n_fft, hop_length, win_length, window_cpu,
            center, normalized, onesided, length, return_complex
        )
        return result.to(input.device)
    return _original_istft(
        input, n_fft, hop_length, win_length, window,
        center, normalized, onesided, length, return_complex
    )


def enable_global_mps_patches():
    """
    Enable global monkey patches for MPS compatibility.
    Call this at the start of your program if you want automatic MPS handling.
    
    Warning: This modifies torch functions globally and may have unintended effects.
    """
    torch.istft = patched_istft
    print("MPS compatibility patches enabled globally")


def disable_global_mps_patches():
    """Disable global monkey patches."""
    torch.istft = _original_istft
    print("MPS compatibility patches disabled")
