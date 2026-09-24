# Master End-to-End model combining all modules
import os
import sys

import librosa
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import AUDIO_SAMPLE_RATE, LLM_DIM
from src.models.decoder import MimiCodec
from src.models.encoder import SpeechEncoder
from src.models.projection import AudioToTextProjection
from src.training.stage3_finetuning import (
    MAX_GENERATED_TOKENS,
    STAGE1_PROJECTOR_PATH,
    STAGE3_LORA_PATH,
    generate_audio_tokens,
    load_stage3_llm,
    select_device,
    trim_to_codes,
    unflatten_ids,
)


class S2SModel(nn.Module):
    """End-to-end speech-to-speech pipeline: 16kHz waveform in -> 24kHz waveform out.
    SpeechEncoder -> Stage-1 projector -> Stage-3 PEFT LLM (autoregressive audio-token
    decoder) -> MimiCodec vocoder. Inference/demo only: batch_size=1.
    """

    def __init__(
        self,
        stage1_projector_path: str = STAGE1_PROJECTOR_PATH,
        stage3_adapter_dir: str = STAGE3_LORA_PATH,
        device: torch.device | None = None,
        max_new_tokens: int = MAX_GENERATED_TOKENS,
        temperature: float = 0.0,
        top_k: int | None = None,
    ):
        super().__init__()
        self.device = device or select_device()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k

        self.encoder = SpeechEncoder(device=self.device, freeze=True)
        self.projector = AudioToTextProjection(
            encoder_dim=self.encoder.hidden_dim, llm_dim=LLM_DIM,
            weights_path=stage1_projector_path, device=self.device, freeze=True,
        ).to(torch.bfloat16)
        self.llm, self.vocab, self.backbone = load_stage3_llm(stage3_adapter_dir, device=self.device)
        self.llm.eval()
        self.codec = MimiCodec(device=self.device, freeze=True)

    @torch.no_grad()
    def speak(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """waveform: 1-D tensor at any sample_rate -> 1-D tensor at MIMI_SAMPLE_RATE (24kHz)."""
        if sample_rate != AUDIO_SAMPLE_RATE:
            waveform = torch.from_numpy(
                librosa.resample(waveform.numpy(), orig_sr=sample_rate, target_sr=AUDIO_SAMPLE_RATE)
            ).float()

        features = self.encoder(waveform.unsqueeze(0))  # [1, T, 512]
        projected = self.projector(features.to(torch.bfloat16))  # [1, T, LLM_DIM]

        token_ids = generate_audio_tokens(
            self.llm, self.vocab, projected,
            max_new_tokens=self.max_new_tokens, temperature=self.temperature, top_k=self.top_k,
        )
        code_ids = trim_to_codes(token_ids, self.vocab)
        if code_ids.numel() == 0:
            raise RuntimeError("Generation produced no complete audio frame.")
        codes = unflatten_ids(code_ids, self.vocab)  # [K, T]
        return self.codec.decode(codes)  # [num_samples] @ 24kHz
