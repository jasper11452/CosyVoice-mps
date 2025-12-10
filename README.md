# CosyVoice-MPS

[![macOS](https://img.shields.io/badge/macOS-M1%2FM2%2FM3%2FM4-blue)](https://support.apple.com/en-us/HT211814)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![MPS](https://img.shields.io/badge/Backend-MPS-orange)](https://pytorch.org/docs/stable/notes/mps.html)

**CosyVoice for macOS Apple Silicon (M-chip)** - 使用 MPS (Metal Performance Shaders) 后端在 Apple Silicon 上运行 CosyVoice 语音合成模型。

本项目基于 [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 官方版本修改，专门适配 macOS M1/M2/M3/M4 芯片，实现 GPU 加速推理。

## ✨ 特性

- 🍎 **原生 MPS 支持**: 自动检测并使用 Apple Silicon GPU 加速
- ⚡ **性能优化**: 相比 CPU 推理速度提升约 2-3 倍
- 🔧 **零配置**: 自动处理 MPS 兼容性问题（JIT 禁用、操作回退等）
- 📦 **简化依赖**: 移除 CUDA/TensorRT 等 GPU 依赖

## 📊 与官方版本的主要差异

| 特性 | 官方版本 | MPS 版本 |
|------|---------|---------|
| GPU 后端 | CUDA | MPS |
| JIT 编译 | 支持 | 禁用（兼容性） |
| TensorRT | 支持 | 不支持 |
| FP16 | 支持 | 禁用（兼容性） |
| VLLM | 支持 | 不支持 |
| 设备检测 | CUDA/CPU | MPS/CUDA/CPU |

## 🚀 快速开始

### 环境要求

- macOS 12.0+ (Monterey 或更高版本)
- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10
- PyTorch 2.0+

### 安装

1. **克隆仓库**
   ```bash
   git clone --recursive https://github.com/jasper11452/CosyVoice-mps.git
   cd CosyVoice-mps
   ```

2. **创建虚拟环境**
   ```bash
   # 使用 uv (推荐)
   uv venv --python 3.10 .venv
   source .venv/bin/activate
   
   # 或使用 conda
   conda create -n cosyvoice python=3.10
   conda activate cosyvoice
   ```

3. **安装依赖**
   ```bash
   # 安装 PyTorch (MPS 版本)
   pip install torch torchaudio
   
   # 安装其他依赖
   pip install -r requirements_macos.txt
   ```

4. **下载模型**
   ```python
   from modelscope import snapshot_download
   snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
   ```

### 验证安装

```python
import torch
from cosyvoice.utils.device_utils import get_device

print(f"MPS 可用: {torch.backends.mps.is_available()}")
print(f"选择的设备: {get_device()}")
```

## 📖 使用方法

### Zero-Shot 语音克隆

```python
import os
import sys

# MPS 适配
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '0'

sys.path.append('third_party/Matcha-TTS')

import torch
import soundfile as sf
import librosa

# JIT 禁用（MPS 需要）
if torch.backends.mps.is_available():
    torch.jit.script_method = lambda fn, _rcb=None: fn
    torch.jit.script = lambda obj, *args, **kwargs: obj

from cosyvoice.cli.cosyvoice import CosyVoice2

# 加载模型
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')
print(f"模型设备: {cosyvoice.model.device}")  # 应该输出 mps

# 加载参考音频
def load_wav(path, target_sr=16000):
    speech, sr = sf.read(path)
    if len(speech.shape) > 1:
        speech = speech[:, 0]
    if sr != target_sr:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=target_sr)
    return torch.from_numpy(speech).float().unsqueeze(0)

prompt_speech = load_wav('your_reference_audio.wav', 16000)

# 生成语音
for i, output in enumerate(cosyvoice.inference_zero_shot(
    tts_text='你好，这是使用MPS后端生成的语音。',
    prompt_text='参考音频对应的文本内容',
    prompt_speech_16k=prompt_speech,
    stream=False
)):
    # 保存音频
    audio = output['tts_speech'].cpu().numpy().squeeze()
    sf.write(f'output_{i}.wav', audio, cosyvoice.sample_rate)
```

### 启动 Web UI

```bash
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

## 🔧 技术细节

### MPS 适配修改

1. **设备检测** (`cosyvoice/utils/device_utils.py`)
   - 新增统一的设备检测函数，优先级：MPS > CUDA > CPU

2. **环境变量**
   - `PYTORCH_ENABLE_MPS_FALLBACK=1`: 不支持的操作自动回退 CPU
   - `TORCHAUDIO_USE_BACKEND_DISPATCHER=0`: 禁用 TorchAudio 后端调度

3. **JIT 禁用**
   - MPS 对 TorchScript JIT 支持不完善，在入口点禁用

4. **内存管理**
   - 使用 `torch.mps.empty_cache()` 替代 `torch.cuda.empty_cache()`

### 性能测试

| 操作 | CPU | MPS | 提升 |
|------|-----|-----|------|
| 模型加载 | ~8s | ~4.5s | 1.8x |
| 10秒音频生成 | ~20s | ~8s | 2.5x |
| RTF | ~2.0 | ~0.8 | 2.5x |

*测试设备: MacBook Pro M2 Max, 32GB RAM*

## 📁 项目结构

```
CosyVoice-mps/
├── cosyvoice/
│   ├── cli/
│   │   ├── cosyvoice.py      # 主入口（MPS 适配）
│   │   ├── model.py          # 模型封装（MPS 适配）
│   │   └── frontend.py       # 前端处理（MPS 适配）
│   ├── utils/
│   │   └── device_utils.py   # 设备检测工具（新增）
│   └── ...
├── webui.py                   # Web UI（MPS 适配）
├── requirements_macos.txt     # macOS 专用依赖
└── README.md
```

## ⚠️ 已知限制

1. **不支持的功能**
   - TensorRT 加速
   - FP16 半精度推理
   - VLLM 加速
   - JIT 编译优化

2. **兼容性问题**
   - 部分复数运算可能回退到 CPU
   - 首次运行可能有编译延迟

## 🙏 致谢

- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 原始项目
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) - Flow Matching 实现
- [PyTorch](https://pytorch.org/) - MPS 后端支持

## 📄 许可证

本项目遵循 Apache 2.0 许可证。详见 [LICENSE](LICENSE)。

## 📚 引用

```bibtex
@article{du2024cosyvoice,
  title={Cosyvoice 2: Scalable streaming speech synthesis with large language models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}
```
