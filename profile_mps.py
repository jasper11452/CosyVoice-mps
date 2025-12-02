#!/usr/bin/env python3
"""MPS 性能分析脚本 - 检查哪些操作在 CPU 上运行"""

import sys
sys.path.append('third_party/Matcha-TTS')

import torch
import time

def check_tensor_device(tensor, name="tensor"):
    """检查张量所在设备"""
    if hasattr(tensor, 'device'):
        print(f"  {name}: {tensor.device}")
    return tensor

def profile_mps_usage():
    """分析 MPS 使用情况"""
    print("=" * 60)
    print("🔍 MPS 性能分析")
    print("=" * 60)
    
    # 1. 检查 MPS 可用性
    print("\n📌 MPS 状态:")
    print(f"  MPS 可用: {torch.backends.mps.is_available()}")
    print(f"  MPS 已构建: {torch.backends.mps.is_built()}")
    
    # 2. 加载模型并检查设备
    print("\n📌 加载模型...")
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', device='mps')
    
    # 3. 检查各模块所在设备
    print("\n📌 模块设备分布:")
    
    def check_module_device(module, name):
        try:
            param = next(module.parameters())
            print(f"  {name}: {param.device}")
        except StopIteration:
            print(f"  {name}: 无参数")
    
    check_module_device(cosyvoice.model.llm, "LLM")
    check_module_device(cosyvoice.model.flow, "Flow")
    check_module_device(cosyvoice.model.hift, "HiFT")
    
    # 4. 检查 HiFT 中的 ISTFT 窗口
    print("\n📌 HiFT STFT 窗口位置:")
    print(f"  stft_window: {cosyvoice.model.hift.stft_window.device}")
    
    # 5. 检查 mps_compat 中的 CPU fallback
    print("\n📌 检查 CPU Fallback 操作:")
    from cosyvoice.utils import mps_compat
    print("  以下操作会 fallback 到 CPU:")
    print("  - torch.istft (STFT 逆变换)")
    print("  - torch.multinomial (采样)")
    
    # 6. 简单推理测试
    print("\n📌 推理性能测试:")
    prompt_speech = load_wav('/Users/jasper/Desktop/1111.wav', 16000)
    
    # 预热
    print("  预热中...")
    for _ in cosyvoice.inference_zero_shot(
        "测试",
        "测试文本",
        prompt_speech,
        stream=False
    ):
        pass
    
    # 同步 MPS
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    # 计时测试
    print("  正式测试...")
    test_text = "你好，这是一段测试语音，用于分析MPS的利用率。"
    
    start = time.time()
    for result in cosyvoice.inference_zero_shot(
        test_text,
        "测试文本内容",
        prompt_speech,
        stream=False
    ):
        pass
    
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    elapsed = time.time() - start
    print(f"  推理耗时: {elapsed:.2f}秒")
    
    # 7. 分析瓶颈
    print("\n📌 潜在瓶颈分析:")
    print("""
  1. LLM 自回归解码: 每个 token 需要单独前向传播
     - 无法批处理，GPU 利用率天然较低
     
  2. CPU Fallback 操作:
     - istft: HiFT 声码器中的 STFT 逆变换
     - 数据需要在 MPS <-> CPU 之间来回传输
     
  3. 小批量推理:
     - batch_size=1 无法充分利用 GPU 并行性
     
  4. 内存带宽:
     - MPS 统一内存架构，CPU/GPU 共享内存
     - 但仍有同步开销
""")

if __name__ == "__main__":
    profile_mps_usage()
