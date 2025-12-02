# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
# MPS-only version - simplified for Apple Silicon inference
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

from typing import Generator
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out


class CosyVoiceModel:
    """CosyVoice Model for MPS inference only."""

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False,
                 device: str = 'mps'):
        self.device = device
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.lock = threading.Lock()
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        self.mel_overlap_len = 34 * 2
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        self.mel_window = torch.from_numpy(self.mel_window).float()
        self.speech_window = np.hamming(2 * 256 * self.mel_overlap_len)
        self.speech_window = torch.from_numpy(self.speech_window).float()

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device, weights_only=True), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device, weights_only=True), strict=True)
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device, weights_only=True), strict=True)
        self.hift.to(self.device).eval()

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        with self.lock:
            for token in self.llm.inference(
                text=text.to(self.device),
                text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                prompt_text=prompt_text.to(self.device),
                prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                embedding=llm_embedding.to(self.device).half() if self.fp16 else llm_embedding.to(self.device),
                uuid=uuid,
            ):
                yield token

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        # MPS inference path
        tts_mel, _ = self.flow.inference(
            token=token.to(self.device),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
            embedding=embedding.to(self.device)
        )
        tts_mel = tts_mel
        
        # speed adjustments
        if speed != 1.0:
            tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            
        tts_speech, _ = self.hift.inference(mel=tts_mel)
        return tts_speech

    def tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        this_uuid = str(uuid.uuid1())

        with torch.no_grad():
            # MPS doesn't support autocast like CUDA
            if stream:
                yield from self._stream_tts(text, flow_embedding, llm_embedding, prompt_text,
                                          llm_prompt_speech_token, flow_prompt_speech_token,
                                          prompt_speech_feat, speed, this_uuid)
            else:
                yield from self._batch_tts(text, flow_embedding, llm_embedding, prompt_text,
                                          llm_prompt_speech_token, flow_prompt_speech_token,
                                          prompt_speech_feat, speed, this_uuid)

    def _stream_tts(self, text, flow_embedding, llm_embedding, prompt_text,
                    llm_prompt_speech_token, flow_prompt_speech_token,
                    prompt_speech_feat, speed, this_uuid):
        # Streaming generation
        token_hop_len = self.token_min_hop_len
        tts_speech_token = []
        tts_speech = torch.zeros(1, 0)
        hift_cache_source = torch.zeros(1, 1, 0)
        
        for token in self.llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid):
            tts_speech_token.append(token)
            if len(tts_speech_token) >= token_hop_len + self.token_overlap_len:
                this_tts_speech_token = torch.tensor(tts_speech_token[:token_hop_len + self.token_overlap_len]).unsqueeze(0)
                this_tts_speech = self.token2wav(
                    token=this_tts_speech_token,
                    prompt_token=flow_prompt_speech_token,
                    prompt_feat=prompt_speech_feat,
                    embedding=flow_embedding,
                    uuid=this_uuid,
                    finalize=False,
                    speed=speed
                )
                if tts_speech.shape[1] != 0:
                    this_tts_speech = fade_in_out(this_tts_speech, tts_speech, self.speech_window.to(self.device))
                yield {'tts_speech': this_tts_speech.cpu()}
                tts_speech_token = tts_speech_token[token_hop_len:]
                tts_speech = this_tts_speech
                token_hop_len = min(self.token_max_hop_len, int(token_hop_len * 1.5))

        # Final tokens
        if len(tts_speech_token) > 0:
            this_tts_speech_token = torch.tensor(tts_speech_token).unsqueeze(0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                uuid=this_uuid,
                finalize=True,
                speed=speed
            )
            if tts_speech.shape[1] != 0:
                this_tts_speech = fade_in_out(this_tts_speech, tts_speech, self.speech_window.to(self.device))
            yield {'tts_speech': this_tts_speech.cpu()}

    def _batch_tts(self, text, flow_embedding, llm_embedding, prompt_text,
                   llm_prompt_speech_token, flow_prompt_speech_token,
                   prompt_speech_feat, speed, this_uuid):
        # Non-streaming: collect all tokens first
        tts_speech_token = []
        for token in self.llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid):
            tts_speech_token.append(token)
            
        tts_speech_token = torch.tensor(tts_speech_token).unsqueeze(0)
        this_tts_speech = self.token2wav(
            token=tts_speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            uuid=this_uuid,
            finalize=True,
            speed=speed
        )
        yield {'tts_speech': this_tts_speech.cpu()}


class CosyVoice2Model(CosyVoiceModel):
    """CosyVoice2 Model for MPS inference only."""

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False,
                 device: str = 'mps'):
        super().__init__(llm, flow, hift, fp16, device)
        self.mel_cache_len = 20
        self.source_cache_len = 20 * 256
        self.token_min_hop_len = self.flow.token_mel_ratio * 3
        self.token_max_hop_len = self.flow.token_mel_ratio * 50

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        tts_mel, _ = self.flow.inference(
            token=token.to(self.device),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
            embedding=embedding.to(self.device),
            streaming=not finalize,
            finalize=finalize
        )
        
        if speed != 1.0:
            tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            
        tts_speech, _ = self.hift.inference(mel=tts_mel)
        return tts_speech
