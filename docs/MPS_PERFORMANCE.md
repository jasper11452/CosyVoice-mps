# CosyVoice MPS 性能优化指南

## 🔍 MPS 利用率低的原因分析

### 主要瓶颈

| 瓶颈 | 时间占比 | 原因 | 可优化性 |
|-----|---------|------|---------|
| **LLM 自回归解码** | ~70% | 每个 token 需单独前向传播，无法批处理 | ❌ 架构限制 |
| **Flow Decoder** | ~20% | Transformer attention 计算 | ⚠️ 有限 |
| **HiFT 声码器** | ~10% | ISTFT 必须在 CPU 运行 | ⚠️ 有限 |

### 具体原因

1. **LLM Token-by-Token 生成**: 自回归模型的固有限制
2. **ISTFT CPU Fallback**: MPS 不支持 `torch.istft`
3. **小批量推理**: batch_size=1 无法充分利用 GPU 并行性

## ⚡ 可用的优化方法

### 1. 模型预热 (推荐)

首次推理会触发 MPS shader 编译，预热可显著提升后续速度：

```python
# 预热模型
for _ in cosyvoice.inference_zero_shot("测试", "测试", prompt_speech, stream=False):
    pass
# 之后的推理会更快
```

### 2. 环境变量优化

```bash
# 启用 MPS 回退（避免不支持的操作报错）
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 禁用 MPS 内存限制（谨慎使用）
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### 3. torch.compile (实验性)

**注意**: PyTorch 2.8+ 才可能稳定支持 MPS compile

```python
# 目前不推荐，可能导致性能下降
# model = torch.compile(model, backend="inductor")
```

## 📊 性能基准

| 指标 | 典型值 |
|-----|-------|
| RTF (实时因子) | 1.0-1.5x |
| 首次推理 | 较慢 (MPS 编译) |
| 后续推理 | 正常速度 |

> RTF 1.3x 表示生成 10 秒音频需要约 13 秒

## 🎯 实际期望

MPS 相比 CUDA 的性能差距主要来自：

1. MPS 生态不如 CUDA 成熟
2. 部分算子需要 fallback 到 CPU
3. Apple Silicon 的 GPU 架构与 NVIDIA 不同
4. torch.compile 对 MPS 支持仍处于早期原型阶段

**合理期望**: MPS 推理速度约为 CUDA 的 30-50%，RTF 约 1.0-1.5x。

## 🔮 未来展望

- PyTorch 2.8 计划改进 torch.compile 对 MPS 的支持
- Apple 持续优化 Metal Performance Shaders
- 社区正在开发更多 MPS 原生算子
