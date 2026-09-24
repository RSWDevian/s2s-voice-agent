# LLM Loader and LoRA Configurations
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure root directory for accessible imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.config import CHECKPOINTS_DIR, DEVICE, LLM_ID


class QwenBackbone(nn.Module):
    """
    Standalone wrapper around the Qwen causal LM used as the reasoning/generation backbone.
    Loads from the local checkpoint cache populated by `src/utils/download_weights.py` when
    available, falling back to the Hugging Face hub otherwise.
    """

    def __init__(self, llm_id: str = LLM_ID, device: str = DEVICE, freeze: bool = True):
        super().__init__()
        self.device = device

        local_backbone = os.path.join(CHECKPOINTS_DIR, "qwen_backbone")
        if os.path.exists(local_backbone) and len(os.listdir(local_backbone)) > 0:
            source, cache_kwargs = local_backbone, {}
            print(f"[*] Loading Qwen backbone from folder: {local_backbone} onto {self.device}...")
        else:
            source, cache_kwargs = llm_id, {"cache_dir": CHECKPOINTS_DIR}
            print(f"[*] Loading Qwen backbone using local cache directory: {CHECKPOINTS_DIR} onto {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(source, **cache_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.bfloat16, attn_implementation="sdpa", **cache_kwargs
        ).to(self.device)

        if freeze:
            self.freeze_weights()

    def freeze_weights(self):
        """Freeze all backbone parameters and set evaluation mode."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        print("[*] Qwen backbone weights frozen for inference / evaluation.")

    def unfreeze_weights(self):
        """Unfreeze all backbone parameters and set training mode."""
        self.train()
        for param in self.parameters():
            param.requires_grad = True
        print("[*] Qwen backbone weights unfrozen for training.")


if __name__ == "__main__":
    # Standalone smoke test
    print(f"[*] Testing QwenBackbone on {DEVICE}...")
    backbone = QwenBackbone()
    print(f"[+] Tokenizer vocab size: {len(backbone.tokenizer)}")
    print(f"[+] Model hidden size: {backbone.model.config.hidden_size}")
