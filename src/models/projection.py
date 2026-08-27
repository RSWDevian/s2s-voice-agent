# MLP modality adapters (Audio <-> LLM Embeddings)
import torch
import torch.nn as nn

class AutioToTextProjection(nn.Module):
    """
    A 2-layer Multi-layer perceptron (MLP) that maps continuous audio embeddings
    into the LLM's semantic vector space.
    """
    def __init__(self, encoder_dim = 1024, llm_dim = 1536, dropout:float = 0.1):
        super().__init__()

        # 2-layer MLP with GeLU activation function
        self.projector = nn.sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using a normal distribution for stable training"""
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, acoustic_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP to project audio embeddings into LLM space.
        Args:
            acoustic_features (torch.Tensor): Input audio embeddings of shape (batch_size, seq_len, encoder_dim)
        Returns:
            torch.Tensor: Projected embeddings of shape (batch_size, seq_len, llm_dim)
        """
        return self.projector(acoustic_features)

if __name__ == "__main__":
    dummy_audio_tensor = torch.randn(2, 150, 1024)
    print("Initializing Audio-to-text Projector...")
    model = AutioToTextProjection(encoder_dim=1024, llm_dim=1536)
    output = model(dummy_audio_tensor)
    print("Input shape:", dummy_audio_tensor.shape)
    print("Output shape:", output.shape)
    print("[*] The Projector successsfully reshaped the audio tensor to the LLM tensor.")
    


