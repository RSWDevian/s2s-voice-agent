# FastConformer wrapper and feature extractor
import os
import sys
import torch
import torch.nn as nn
from nemo.collections.asr.models import EncDecCTCModel

# Ensure root directory for accessible imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.config import CHECKPOINTS_DIR, DEVICE

class SpeechEncoder(nn.Module):
    """
    Standalone Speech Encoder wrapping NVIDIA's FastConformer.
    Processes raw 16kHz audio waveforms and extracts continous acoustic
    representations for downstream projection into LLM.
    """

    def __init__(
        self,
        model_name: str = "nvidia/stt_en_fastconformer_hybrid_large_pc",
        device: str = DEVICE,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device

        # Configure local nemo cache directory
        nemo_cache = os.path.join(CHECKPOINTS_DIR, "nemo_cache")
        os.makedirs(nemo_cache, exist_ok=True)
        os.environ["NEMO_CACHE_DIR"] = nemo_cache

        print(f"[*] Loading FastConformer ASR model: {self.model_name} onto {self.device}")
        self.asr_model = EncDecCTCModel.from_pretrained(model_name=self.model_name)
        self.asr_model.to(self.device)

        # Detect encoder hidden dimension
        if hasattr(self.asr_model.encoder, "_feat_out"):
            self.hidden_dim = self.asr_model.encoder._feat_out
        elif hasattr(self.asr_model.encoder, "_d_dim"):
            self.hidden_dim = self.asr_model.encoder._d_dim
        else:
            self.hidden_dim = 512

        print(f"[*] Acoustic encoder output dimension: {self.hidden_dim}")

        if freeze:
            self.freeze_weights()

    def freeze_weights(self):
        """Freeze all encoder parameters and set evaluation mode"""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        print("[*] Encoder weights frozen for inference / evaluation.")

    def unfreeze_weights(self):
        """Unfreeze all encoder parameters and set training mode"""
        self.train()
        for param in self.parameters():
            param.requires_grad = True
        print("[*] Encoder weights unfrozen for training.")

    def forward(
        self,
        audio_signals: torch.Tensor,
        signal_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Extracts continuous latent representations from raw 16kHz signals

        Args:
            audio_signals (torch.Tensor): Audio waveform tensor of shape (batch_size, num_samples)
            signal_lengths (torch.Tensor, Optional): Exact same length of each audio item in batch.
                                                    Defaults to full length is None.
        Returns:
            torch.Tensor: Acoustic latent representations of shape (batch_size, seq_len, hidden_dim)
        """
        audio_signals = audio_signals.to(self.device)

        if signal_lengths is None:
            batch_size, num_samples = audio_signals.shape
            signals_length = torch.full(
                (batch_size,), num_samples, dtype=torch.int32, device=self.device
            )
        else:
            signals_length = signal_lengths.to(self.device)

        # 1. Preprocess audio into log-mel spectrogram features
        processed_signal, processed_lengths = self.asr_model.preprocessor(
            input_signal=audio_signals,
            length=signals_length,
        )

        # 2. Extract continuous acoustic features through the Conformer encoder
        encoded, _ = self.asr_model.encoder(
            audio_signal=processed_signal,
            length=processed_lengths,
        )

        # NeMo outputs (batch_size, hidden_dim, seq_len) -> Transpose to (batch_size, seq_len, hidden_dim)
        encoded = encoded.transpose(1, 2)
        return encoded


if __name__ == "__main__":
    # Standalone smoke test
    print(f"[*] Testing SpeechEncoder on {DEVICE}...")
    encoder = SpeechEncoder()

    # Generate 2 dummy 3-second audio clips (16kHz * 3s = 48,000 samples)
    sample_rate = 16000
    duration_sec = 3
    dummy_audio = torch.randn(2, sample_rate * duration_sec)
    dummy_lengths = torch.tensor([sample_rate * duration_sec, sample_rate * 2], dtype=torch.long)

    with torch.no_grad():
        features = encoder(dummy_audio, dummy_lengths)

    print(f"[+] Input audio shape:    {dummy_audio.shape}")
    print(f"[+] Output feature shape: {features.shape}")
    print(f"[+] Features shape verified: (Batch, Seq_Len, {encoder.hidden_dim})")