# Codec/Vocoder wrapper
import os
import sys
import torch
import torch.nn as nn
from transformers import MimiModel

# Ensure root directory for accessible imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.config import CHECKPOINTS_DIR, DEVICE


class MimiCodec(nn.Module):
    """
    Standalone wrapper around Kyutai's Mimi neural audio codec.
    Encodes raw 24kHz waveforms into discrete audio tokens (targets for
    Stage 3 training). Weights are loaded from the local cache populated
    by `src/utils/download_weights.py`.
    """

    def __init__(self, device: str = DEVICE, freeze: bool = True):
        super().__init__()
        self.device = device

        local_path = os.path.join(CHECKPOINTS_DIR, "mimi_codec")
        if not os.path.isdir(local_path):
            raise FileNotFoundError(
                f"Mimi codec weights not found at {local_path}. Run src/utils/download_weights.py first."
            )

        print(f"[*] Loading Mimi codec from {local_path} onto {self.device}")
        self.model = MimiModel.from_pretrained(local_path)
        self.model.to(self.device)

        if freeze:
            self.freeze_weights()

    def freeze_weights(self):
        """Freeze all codec parameters and set evaluation mode."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        print("[*] Mimi codec weights frozen for inference / evaluation.")

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, num_quantizers: int = 8) -> torch.Tensor:
        """
        Encodes a raw waveform sampled at MIMI_SAMPLE_RATE (24kHz) into discrete codes.

        Args:
            waveform (torch.Tensor): Shape (num_samples,) or (1, num_samples).
            num_quantizers (int): Number of RVQ codebooks to keep (Mimi's full stack has
                32, coarse-to-fine; Stage 3 only trains on the first 8, so we ask the
                codec to skip the rest instead of computing/storing all 32).
        Returns:
            torch.Tensor: Mimi codes of shape (num_quantizers, num_frames).
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.unsqueeze(0).to(self.device)  # (1, 1, num_samples)

        output = self.model.encode(waveform, num_quantizers=num_quantizers)
        return output.audio_codes.squeeze(0).cpu()  # (num_quantizers, num_frames)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decodes discrete Mimi codes back into a raw waveform at MIMI_SAMPLE_RATE (24kHz).

        Args:
            codes (torch.Tensor): Mimi codes, shape (num_quantizers, num_frames) or
                (1, num_quantizers, num_frames). num_quantizers should match what the
                codes were encoded with (Stage 3 trains on 8).
        Returns:
            torch.Tensor: Reconstructed waveform, shape (num_samples,), on CPU.
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)  # (1, num_quantizers, num_frames)
        codes = codes.to(self.device).long()

        output = self.model.decode(codes)
        return output.audio_values.squeeze(0).squeeze(0).cpu()  # (num_samples,)


if __name__ == "__main__":
    # Standalone smoke test
    print(f"[*] Testing MimiCodec on {DEVICE}...")
    codec = MimiCodec()

    sample_rate = 24000
    duration_sec = 1
    dummy_audio = torch.randn(sample_rate * duration_sec)

    codes = codec.encode(dummy_audio)
    print(f"[+] Input audio shape: {dummy_audio.shape}")
    print(f"[+] Output codes shape: {codes.shape}")

    waveform = codec.decode(codes)
    print(f"[+] Decoded waveform shape: {waveform.shape}")
