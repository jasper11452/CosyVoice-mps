import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
import time
from datetime import datetime
from pathlib import Path

# ============ 配置区 ============
OUTPUT_DIR = Path("output")
PROMPT_AUDIO = "/Users/jasper/Desktop/1111.wav"
PROMPT_TEXT = "我始终认为，爱一个人就应该只爱一个人。如果爱上其他人，那就是不爱。"

TEXT = """
你要是现在站在我旁边，正好能看见——天快黑了，荷塘边上，一叶小木船，歪歪地浮在水中央，水都快绿成翡翠了，晃得人眼晕。船头坐着个姑娘，穿件白纱裙，飘得跟雾似的，风吹过来，裙摆一抖，好像随时能飞走。她头发黑得发亮，长到腰，一缕一缕的，像墨泼出来的一样，发丝上还夹着几颗小银珠，一动就闪，像天上掉下来的星星，贴着她头发晃。
"""

# 保存独立片段（调试用）
SAVE_SEGMENTS = False
# ===============================

def main():
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("🎤 CosyVoice TTS 语音合成")
    print("=" * 60)
    print(f"📝 文本长度: {len(TEXT.strip())} 字符")
    print(f"🎯 提示音频: {PROMPT_AUDIO}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    # 加载模型
    print("\n⏳ 正在加载 CosyVoice2 模型...")
    start_load = time.time()
    cosyvoice = CosyVoice2(
        'pretrained_models/CosyVoice2-0.5B',
        device='mps'
    )
    print(f"✅ 模型加载完成! 耗时: {time.time() - start_load:.2f}秒\n")
    
    # 加载提示音频
    print(f"⏳ 正在加载提示音频...")
    prompt_speech_16k = load_wav(PROMPT_AUDIO, 16000)
    print(f"✅ 提示音频加载完成!\n")
    
    # 生成语音
    print("🎙️  开始生成语音...\n")
    start_inference = time.time()
    
    audio_segments = []
    segment_count = 0
    
    for i, result in enumerate(cosyvoice.inference_zero_shot(
        TEXT.strip(),
        PROMPT_TEXT,
        prompt_speech_16k,
        stream=False
    )):
        segment_count += 1
        tts_speech = result['tts_speech']
        audio_segments.append(tts_speech)
        
        # 计算音频时长
        duration = tts_speech.shape[1] / cosyvoice.sample_rate
        
        print(f"  ✓ 片段 {segment_count}: {duration:.2f}秒")
        
        # 保存独立片段（如果需要）
        if SAVE_SEGMENTS:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')[:17]
            segment_path = OUTPUT_DIR / f"segment_{segment_count:02d}_{timestamp}.wav"
            torchaudio.save(str(segment_path), tts_speech, cosyvoice.sample_rate)
    
    # 合并所有音频片段
    print(f"\n⏳ 正在合并 {segment_count} 个音频片段...")
    merged_audio = torch.cat(audio_segments, dim=1)
    total_duration = merged_audio.shape[1] / cosyvoice.sample_rate
    
    # 保存合并后的音频
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"{timestamp}.wav"
    output_path = OUTPUT_DIR / output_filename
    
    torchaudio.save(str(output_path), merged_audio, cosyvoice.sample_rate)
    
    inference_time = time.time() - start_inference
    rtf = inference_time / total_duration
    
    print("\n" + "=" * 60)
    print("✅ 语音生成完成!")
    print("=" * 60)
    print(f"📄 输出文件: {output_path}")
    print(f"🎵 音频时长: {total_duration:.2f}秒")
    print(f"⏱️  生成耗时: {inference_time:.2f}秒")
    print(f"📊 实时因子 (RTF): {rtf:.2f}x")
    print(f"🔊 采样率: {cosyvoice.sample_rate} Hz")
    print(f"📦 片段数量: {segment_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()
