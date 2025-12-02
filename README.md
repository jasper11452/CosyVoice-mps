# CosyVoice MPS

CosyVoice 语音合成的 **Apple Silicon (MPS)** 精简版本。

移除了所有 CUDA/TensorRT/VLLM 和训练相关代码，专注于 Mac M1/M2/M3/M4 上的推理。

## 特性

- 🍎 Apple Silicon (MPS) 原生支持
- 🎤 零样本声音克隆
- 🌍 跨语言合成
- ⚡ CosyVoice2 支持

## 环境要求

- macOS + Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- PyTorch 2.0+

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-repo/CosyVoice-mps
cd CosyVoice-mps

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install -e third_party/Matcha-TTS
```

## 下载模型

从 ModelScope 下载预训练模型：

```bash
mkdir -p pretrained_models
# 下载 CosyVoice2-0.5B: https://modelscope.cn/models/iic/CosyVoice2-0.5B
```

## 使用方法

### Python API

```python
import sys
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# 加载模型
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', device='mps')

# 加载参考音频 (16kHz)
prompt_speech = load_wav('your_prompt.wav', 16000)

# 生成语音
for result in cosyvoice.inference_zero_shot(
    tts_text="你好，这是测试语音。",
    prompt_text="参考音频对应的文本内容。",
    prompt_speech_16k=prompt_speech,
    stream=False
):
    torchaudio.save('output.wav', result['tts_speech'], cosyvoice.sample_rate)
```

### 快速测试

```bash
python main.py
```

## 项目结构

```
CosyVoice-mps/
├── cosyvoice/
│   ├── cli/           # 主接口
│   ├── flow/          # Flow matching 解码器
│   ├── hifigan/       # HiFiGAN 声码器
│   ├── llm/           # 语言模型
│   └── utils/         # 工具函数 & MPS 兼容层
├── third_party/
│   └── Matcha-TTS/    # Matcha-TTS 子模块
├── pretrained_models/ # 预训练模型
├── main.py            # 示例脚本
└── requirements.txt
```

## MPS 适配说明

针对 Apple Silicon 做了以下适配：

- `mps_compat.py`: MPS 兼容层，处理不支持的操作 (如 `istft`) 自动 fallback 到 CPU
- `mps_attention_patch.py`: 修复 attention 计算中的数值稳定性问题
- 移除所有 CUDA 特定代码和依赖
- 精简配置文件，移除训练相关组件

## 性能说明

### MPS 利用率

MPS 利用率相对较低是正常现象，主要原因：

1. **LLM 自回归解码** (~70% 时间): 每个 token 需单独前向传播，无法批处理
2. **ISTFT CPU Fallback**: MPS 不支持 `torch.istft`，需回退到 CPU
3. **小批量推理**: batch_size=1 无法充分利用 GPU 并行性

### 性能基准

| 指标 | 典型值 |
|-----|-------|
| RTF (实时因子) | 1.0-1.5x |
| 首次推理 | 较慢 (MPS 编译) |
| 后续推理 | 正常速度 |

> RTF 1.3x 表示生成 10 秒音频需要约 13 秒

### 优化建议

- 预热模型（首次推理会触发 MPS shader 编译）
- 使用较短的文本分段
- 确保系统有足够内存

## 已知问题

- 首次推理较慢（MPS 编译）
- 部分操作回退到 CPU 以保证稳定性
- MPS 整体性能约为 CUDA 的 30-50%

## 许可证

Apache License 2.0

## 致谢

基于 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) by Alibaba.
