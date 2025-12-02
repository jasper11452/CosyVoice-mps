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

她一只手撑着一把旧白油纸伞，伞有点旧，边角都卷了，可她撑得特别稳，伞面微微往一边歪，正好给她挡着光，也挡着风。她回头看了眼，眼神软得不行，嘴角轻轻一勾，不笑，但看着你，好像在说：“你来了啊。”那眼神，温的，像春天的水，你不说话，她也不急，就那么静静看着你，好像你走了她也不难过，可你要是真走了，她心里准得空一下。

背景就是荷塘，一大片荷叶，绿得发黑，铺得老远，叶子上还挂着水珠，一晃一晃的，粉荷花零零星星开着，有的快谢了，花瓣掉在水里，打着转，像没人要的信纸，飘着飘着就没了。天快黑了，但还有点光，从西边照过来，穿过荷叶缝隙，照在她头发上、衣袖上，白纱一下子变得发亮，像裹了层光，整个人都像浮在水面上，不踏实，也不像真的。

她坐着不动，手还捏着伞骨，指节有点发白，好像在忍着什么，又好像什么都不想管了。脚边放着个竹篮，里面扔着本旧书，书角都翻卷了，估计她翻过好多遍，早就背下来了，可还爱翻。

一只蜻蜓飞过来，停在伞尖上，翅膀一闪一闪。她也没动，也不赶，就那么看着水对面，眼神飘得远，像是在等谁，又像是已经等了好久，早就不抱希望了。

风一吹，她发丝飘起来，那几颗银珠也晃了晃，她忽然伸手，从发里摸出一根玉簪，轻轻往头发里一别，动作特别轻，跟做梦似的。

然后她就没动了，也不回头，也不说话。

船慢慢漂，往塘中央去，水声轻轻的，像谁在小声说话。远处有灯，忽明忽暗，像是谁家还没关的门，透出一点暖光。

她坐在那儿，像一幅画，可你又知道——她不是画，她是真的，她就在那儿，风吹她头发，水碰她脚，她就那么静静坐着，不走，也不说话。

你要是过去，她大概也不会抬头，可你要是走，她心里，大概会空一下。

那片荷塘，安静得吓人，可她坐着，像在等一个永远不会来的消息，又像在等一个早就该结束的梦。

她不说话，可你总觉得她心里在念着谁，念着哪句没说完的话，念着哪天没走成的路。

她手还搭在伞上，指甲是淡粉色的，看着不大用力，可你要是伸手，会发现她手指有点凉，像水底捞上来的。

她头微微低着，光从她肩头照过来，照得她衣袖发亮，像穿了层光做的衣服。

你要是蹲下，能看到水面——她倒影也坐在那儿，一模一样，可她倒影的嘴角，会比她真的那一边，多笑一点。

你突然就懂了：她不是在等谁，她只是不肯走，不敢走，也不愿意走。

她怕一走，这光，这水，这风，这梦，就全没了。

她坐在那儿，像一块不肯融化的冰，又像一个还没写完的句号，挂在天和水中间，悬着，悬着，悬着——直到天完全黑，直到风停，直到你忘了她，她才可能真的，走掉。
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
        load_jit=False,
        load_trt=False,
        fp16=False,
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
