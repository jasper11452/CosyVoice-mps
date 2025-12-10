# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# MPS Support for macOS M-chip added
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def get_device():
    """
    获取最佳可用设备。
    优先顺序：MPS > CUDA > CPU
    
    Returns:
        torch.device: 最佳可用设备
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_device_name():
    """
    返回设备名称字符串
    
    Returns:
        str: 设备名称 ('mps', 'cuda', or 'cpu')
    """
    return str(get_device())


def is_mps_available():
    """
    检查 MPS 是否可用
    
    Returns:
        bool: MPS 是否可用
    """
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def is_cuda_available():
    """
    检查 CUDA 是否可用
    
    Returns:
        bool: CUDA 是否可用
    """
    return torch.cuda.is_available()


def empty_cache():
    """
    清空 GPU 缓存（支持 CUDA 和 MPS）
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # MPS 没有显式的 empty_cache，但可以通过 gc 触发
    elif is_mps_available():
        import gc
        gc.collect()
        torch.mps.empty_cache()
