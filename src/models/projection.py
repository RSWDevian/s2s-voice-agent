import os
import torch
import torch.nn as nn
from src.config import DEVICE


class AudioToTextProjection(nn.Module):
    """
    Standalone 2-layer Multi-Layer Perceptron (MLP) Projection Adapter.
    Bridges the gap between continuous acoustic encoder representations (e.g., FastConformer)
    and the semantic embedding space of a Large Language Model (e.g., Qwen).
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        llm_dim: int = 1536,
        dropout: float = 0.1,
        weights_path: str | None = None,
        device: str = DEVICE,
        freeze: bool = False,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim

        # 2-layer Non-linear Projection Architecture
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
        )

        self._init_weights()
        self.to(device)

        if weights_path is not None:
            self.load_weights(weights_path, device=device, freeze=freeze)

    def _init_weights(self):
        """Initializes projection layers with normal distribution for training stability."""
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, acoustic_features: torch.Tensor) -> torch.Tensor:
        """
        Maps acoustic latent sequences into the LLM semantic vector space.

        Args:
            acoustic_features (torch.Tensor): Audio features with shape (batch_size, seq_len, encoder_dim)

        Returns:
            torch.Tensor: Projected embeddings with shape (batch_size, seq_len, llm_dim)
        """
        return self.projector(acoustic_features)

    def load_weights(self, weights_path: str, device: str = "cpu", freeze: bool = True):
        """
        Loads pre-trained state dictionary weights into the projection layer.

        Args:
            weights_path (str): Filepath to the saved .pth checkpoint.
            device (str): Device to map the loaded tensor weights onto.
            freeze (bool): If True, sets requires_grad=False and toggles eval() mode for inference.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"[!] Checkpoint file not found: {weights_path}")

        state_dict = torch.load(weights_path, map_location=device)
        self.load_state_dict(state_dict)
        print(f"[*] Successfully loaded projector weights from: {weights_path}")

        if freeze:
            self.freeze_weights()

    def freeze_weights(self):
        """Switches model to eval mode and stops tracking gradients for inference."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        print("[*] Projector weights frozen for inference / evaluation.")

    def unfreeze_weights(self):
        """Restores model to train mode and enables gradient updates."""
        self.train()
        for param in self.parameters():
            param.requires_grad = True
        print("[*] Projector weights unfrozen for training.")

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str,
        encoder_dim: int = 512,
        llm_dim: int = 1536,
        device: str = "cpu",
        freeze: bool = True,
    ) -> "AudioToTextProjection":
        """
        Factory helper to initialize and load trained weights in a single call.
        """
        return cls(
            encoder_dim=encoder_dim,
            llm_dim=llm_dim,
            weights_path=weights_path,
            device=device,
            freeze=freeze,
        )


if __name__ == "__main__":
    # Smoke test for standalone initialization and projection
    test_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running standalone test on device: {test_device}")

    # 1. Initialize empty architecture
    mlp = AudioToTextProjection(encoder_dim=512, llm_dim=1536, device=test_device)
    dummy_input = torch.randn(2, 150, 512, device=test_device)

    output = mlp(dummy_input)
    print(f"[+] Input shape:  {dummy_input.shape}")
    print(f"[+] Output shape: {output.shape}")

    assert output.shape == (2, 150, 1536), "Output shape mismatch!"
    print("[+] Projection test passed successfully.")