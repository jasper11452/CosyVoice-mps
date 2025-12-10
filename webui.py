#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CosyVoice MPS Web UI
适用于 macOS Apple Silicon (M1/M2/M3/M4)
"""
import os
import sys

# ============== MPS (Apple Silicon) 适配 ==============
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '0'
# ======================================================

import argparse
import gradio as gr
import numpy as np
import torch
import soundfile as sf
import librosa
import random

# JIT 禁用（MPS 兼容性）
if torch.backends.mps.is_available():
    torch.jit.script_method = lambda fn, _rcb=None: fn
    torch.jit.script = lambda obj, *args, **kwargs: obj
    print("✅ MPS 检测到，JIT 已禁用")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'third_party/Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.common import set_all_random_seed

# ============== 配置 ==============
MAX_VAL = 0.8
SAMPLE_RATE = 24000

# ============== 工具函数 ==============

def load_wav(wav_path, target_sr=16000):
    """使用 soundfile 加载音频并重采样"""
    speech, sr = sf.read(wav_path)
    if len(speech.shape) > 1:
        speech = speech[:, 0]  # 取第一个声道
    if sr != target_sr:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=target_sr)
    return torch.from_numpy(speech).float().unsqueeze(0)


def get_audio_info(wav_path):
    """获取音频信息"""
    info = sf.info(wav_path)
    return info.samplerate, info.duration


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """后处理音频"""
    speech_np = speech.cpu().numpy().squeeze()
    speech_np, _ = librosa.effects.trim(
        speech_np, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    speech = torch.from_numpy(speech_np).unsqueeze(0)
    if speech.abs().max() > MAX_VAL:
        speech = speech / speech.abs().max() * MAX_VAL
    # 添加尾部静音
    silence = torch.zeros(1, int(SAMPLE_RATE * 0.2))
    speech = torch.cat([speech, silence], dim=1)
    return speech


def generate_seed():
    """生成随机种子"""
    return random.randint(1, 100000000)


# ============== 主生成函数 ==============

def generate_audio(
    tts_text,
    mode,
    sft_speaker,
    prompt_text,
    prompt_wav_upload,
    prompt_wav_record,
    instruct_text,
    seed,
    speed
):
    """生成音频"""
    if not tts_text.strip():
        yield None, "请输入要合成的文本"
        return
    
    set_all_random_seed(seed)
    
    # 确定参考音频
    prompt_wav = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
    
    try:
        if mode == "预训练音色":
            # SFT 模式
            if not sft_speaker:
                yield None, "请选择预训练音色"
                return
            
            result = []
            for output in cosyvoice.inference_sft(tts_text, sft_speaker, stream=False):
                result.append(output['tts_speech'])
            
            if result:
                speech = torch.cat(result, dim=1)
                speech = postprocess(speech)
                audio_np = speech.cpu().numpy().squeeze()
                yield (SAMPLE_RATE, audio_np), "✅ 生成成功"
            else:
                yield None, "生成失败：没有输出"
                
        elif mode == "3s极速复刻":
            # Zero-shot 模式
            if not prompt_wav:
                yield None, "请上传或录制参考音频"
                return
            if not prompt_text.strip():
                yield None, "请输入参考音频对应的文本"
                return
            
            # 检查音频时长
            sr, duration = get_audio_info(prompt_wav)
            if duration > 30:
                yield None, "参考音频不能超过30秒"
                return
            
            prompt_speech_16k = load_wav(prompt_wav, 16000)
            
            result = []
            for output in cosyvoice.inference_zero_shot(
                tts_text, prompt_text, prompt_speech_16k, stream=False
            ):
                result.append(output['tts_speech'])
            
            if result:
                speech = torch.cat(result, dim=1)
                speech = postprocess(speech)
                audio_np = speech.cpu().numpy().squeeze()
                yield (SAMPLE_RATE, audio_np), "✅ 生成成功"
            else:
                yield None, "生成失败：没有输出"
                
        elif mode == "跨语种复刻":
            # Cross-lingual 模式
            if not prompt_wav:
                yield None, "请上传或录制参考音频"
                return
            
            sr, duration = get_audio_info(prompt_wav)
            if duration > 30:
                yield None, "参考音频不能超过30秒"
                return
            
            prompt_speech_16k = load_wav(prompt_wav, 16000)
            
            result = []
            for output in cosyvoice.inference_cross_lingual(
                tts_text, prompt_speech_16k, stream=False
            ):
                result.append(output['tts_speech'])
            
            if result:
                speech = torch.cat(result, dim=1)
                speech = postprocess(speech)
                audio_np = speech.cpu().numpy().squeeze()
                yield (SAMPLE_RATE, audio_np), "✅ 生成成功"
            else:
                yield None, "生成失败：没有输出"
                
        elif mode == "自然语言控制":
            # Instruct 模式 (需要 instruct 模型或使用 instruct2)
            if not prompt_wav:
                yield None, "请上传或录制参考音频"
                return
            if not instruct_text.strip():
                yield None, "请输入指令文本"
                return
            
            sr, duration = get_audio_info(prompt_wav)
            if duration > 30:
                yield None, "参考音频不能超过30秒"
                return
            
            prompt_speech_16k = load_wav(prompt_wav, 16000)
            
            result = []
            for output in cosyvoice.inference_instruct2(
                tts_text, instruct_text, prompt_speech_16k, stream=False
            ):
                result.append(output['tts_speech'])
            
            if result:
                speech = torch.cat(result, dim=1)
                speech = postprocess(speech)
                audio_np = speech.cpu().numpy().squeeze()
                yield (SAMPLE_RATE, audio_np), "✅ 生成成功"
            else:
                yield None, "生成失败：没有输出"
        else:
            yield None, f"未知模式: {mode}"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"❌ 生成失败: {str(e)}"


def update_ui(mode):
    """根据模式更新 UI 可见性"""
    if mode == "预训练音色":
        return (
            gr.update(visible=True),   # sft_speaker
            gr.update(visible=False),  # prompt_text
            gr.update(visible=False),  # prompt_wav_upload
            gr.update(visible=False),  # prompt_wav_record
            gr.update(visible=False),  # instruct_text
        )
    elif mode == "3s极速复刻":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    elif mode == "跨语种复刻":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    elif mode == "自然语言控制":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        )


# ============== 主程序 ==============

def main():
    global cosyvoice, SAMPLE_RATE
    
    parser = argparse.ArgumentParser(description='CosyVoice MPS Web UI')
    parser.add_argument('--port', type=int, default=50000, help='服务端口')
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B',
                        help='模型目录')
    parser.add_argument('--share', action='store_true', help='创建公开链接')
    args = parser.parse_args()
    
    # 加载模型
    print(f"正在加载模型: {args.model_dir}")
    print(f"设备: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    
    cosyvoice = CosyVoice2(args.model_dir)
    SAMPLE_RATE = cosyvoice.sample_rate
    
    print(f"✅ 模型加载完成，设备: {cosyvoice.model.device}")
    
    # 获取可用音色
    available_spks = cosyvoice.list_available_spks()
    print(f"可用音色: {available_spks}")
    
    # 创建 Gradio 界面
    with gr.Blocks(title="CosyVoice MPS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎤 CosyVoice MPS
        **在 Apple Silicon 上运行的语音合成模型**
        
        支持模式：预训练音色、3s极速复刻、跨语种复刻、自然语言控制
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 模式选择
                mode = gr.Radio(
                    choices=["预训练音色", "3s极速复刻", "跨语种复刻", "自然语言控制"],
                    value="3s极速复刻",
                    label="合成模式"
                )
                
                # 预训练音色选择
                sft_speaker = gr.Dropdown(
                    choices=available_spks,
                    value=available_spks[0] if available_spks else None,
                    label="预训练音色",
                    visible=False
                )
                
                # 合成文本
                tts_text = gr.Textbox(
                    label="合成文本",
                    placeholder="请输入要合成的文本...",
                    lines=3
                )
                
                # 参考音频文本
                prompt_text = gr.Textbox(
                    label="参考音频文本",
                    placeholder="请输入参考音频对应的文本内容...",
                    lines=2,
                    visible=True
                )
                
                # 参考音频上传
                prompt_wav_upload = gr.Audio(
                    label="上传参考音频",
                    type="filepath",
                    visible=True
                )
                
                # 参考音频录制
                prompt_wav_record = gr.Audio(
                    label="录制参考音频",
                    sources=["microphone"],
                    type="filepath",
                    visible=True
                )
                
                # 指令文本
                instruct_text = gr.Textbox(
                    label="指令文本",
                    placeholder="例如：用四川话说这句话",
                    visible=False
                )
                
                with gr.Row():
                    seed = gr.Number(label="随机种子", value=42, precision=0)
                    seed_btn = gr.Button("🎲 随机", size="sm")
                
                speed = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                    label="语速"
                )
                
                generate_btn = gr.Button("🎵 生成音频", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_audio = gr.Audio(label="生成的音频", type="numpy")
                output_text = gr.Textbox(label="状态", interactive=False)
        
        # 事件绑定
        mode.change(
            fn=update_ui,
            inputs=[mode],
            outputs=[sft_speaker, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text]
        )
        
        seed_btn.click(
            fn=generate_seed,
            outputs=[seed]
        )
        
        generate_btn.click(
            fn=generate_audio,
            inputs=[
                tts_text, mode, sft_speaker, prompt_text,
                prompt_wav_upload, prompt_wav_record, instruct_text,
                seed, speed
            ],
            outputs=[output_audio, output_text]
        )
    
    # 启动服务
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
