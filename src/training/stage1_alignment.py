import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod

# Ensuring root directory is available for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import CHECKPOINTS_DIR, DEVICE

try:
    from src.dataset.dataset import FastConformerDataset
except ModuleNotFoundError:
    from src.dataset.dataset import FastConformerDataset

from src.models.projection import AudioToTextProjection

# Global Configuration
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-4


def pad_audio_collate_fn(batch):
    """
    Pads variable-length audio tensors along the time dimension (dim=0)
    to match the maximum sequence length within the current batch.
    """
    audio_tensors, text_transcripts = zip(*batch)
    # Pads tensors to shape: (Batch_Size, Max_Seq_Len, Encoder_Dim)
    padded_audio = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)
    return padded_audio, list(text_transcripts)


class BaseAlignmentTrainer(ABC):
    """
    Abstract Base Class for training the audio-to-text projection layer.
    """
    def __init__(
        self, 
        batch_size: int = BATCH_SIZE, 
        epochs: int = EPOCHS, 
        learning_rate: float = LEARNING_RATE, 
        resume: bool = False
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        # 1. Load Dataset with dynamic batch padding
        try:
            self.dataset = FastConformerDataset()
            self.dataloader = DataLoader(
                self.dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                collate_fn=pad_audio_collate_fn
            )
        except FileNotFoundError as e:
            print(f"[!] Dataset not found. Ensure manifest.pt exists in processed tensors directory.")
            raise e

        # Detect encoder dimension dynamically from the first sample
        sample_audio, _ = self.dataset[0]
        encoder_dim = sample_audio.shape[-1]
        print(f"[*] Detected acoustic feature dimension: {encoder_dim}")

        # 2. Initialize the Trainable MLP Projector
        print("[*] Initializing the MLP projector...")
        self.projector = AudioToTextProjection(encoder_dim=encoder_dim, llm_dim=1536).to(DEVICE)
        
        # Resume Checkpoint Logic
        save_dir = os.path.join(CHECKPOINTS_DIR, "trained_projector")
        self.checkpoint_path = os.path.join(save_dir, f"mlp_stage1_{self.__class__.__name__}.pth")
        
        if resume:
            if os.path.exists(self.checkpoint_path):
                print(f"[*] Found existing checkpoint. Resuming weights from {self.checkpoint_path}...")
                self.projector.load_state_dict(torch.load(self.checkpoint_path, map_location=DEVICE))
            else:
                print(f"[!] No checkpoint found at {self.checkpoint_path}. Starting from scratch.")
        
        self.projector.train()

        # 3. Setup Optimizer and Loss Function
        self.optimizer = torch.optim.AdamW(self.projector.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # 4. Load the LLM
        self._load_llm()

    @abstractmethod
    def _load_llm(self):
        pass

    @abstractmethod
    def _get_text_embeddings(self, batch_texts: list[str], max_length: int) -> torch.Tensor:
        pass

    def train(self):
        """Universal Stage 1 training loop."""
        print("\n[*] Commencing Training Loop...")
        for epoch in range(self.epochs):
            total_loss = 0.0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch_audio, batch_texts in progress_bar:
                batch_audio = batch_audio.to(DEVICE)
                
                # Forward Pass
                projected_audio = self.projector(batch_audio)
                
                # Target LLM Embeddings
                target_text_embeds = self._get_text_embeddings(
                    batch_texts, 
                    max_length=projected_audio.shape[1]
                ).to(torch.float32)  # Ensure same dtype for loss computation
                
                # Align sequence lengths
                min_seq_len = min(projected_audio.shape[1], target_text_embeds.shape[1])
                projected_audio_aligned = projected_audio[:, :min_seq_len, :]
                target_text_embeds_aligned = target_text_embeds[:, :min_seq_len, :]

                # Optimization
                loss = self.loss_fn(projected_audio_aligned, target_text_embeds_aligned)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                
            avg_loss = total_loss / len(self.dataloader)
            print(f"\n[+] Epoch {epoch+1} Completed. Average Loss: {avg_loss:.4f}")

            self.save_checkpoint()

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(self.projector.state_dict(), self.checkpoint_path)
        print(f"\n[+] Projector weights safely saved to {self.checkpoint_path}")


class QwenAlignmentTrainer(BaseAlignmentTrainer):
    def __init__(self, llm_id: str = "Qwen/Qwen2.5-1.5B", **kwargs):
        self.llm_id = llm_id
        super().__init__(**kwargs)

    def _load_llm(self):
        local_backbone = os.path.join(CHECKPOINTS_DIR, "qwen_backbone")
        
        if os.path.exists(local_backbone) and len(os.listdir(local_backbone)) > 0:
            target_source = local_backbone
            cache_kwargs = {}
            print(f"[*] Loading pre-trained Qwen model directly from folder: {local_backbone} onto {DEVICE}...")
        else:
            target_source = self.llm_id
            cache_kwargs = {"cache_dir": CHECKPOINTS_DIR}
            print(f"[*] Loading pre-trained Qwen model using local cache directory: {CHECKPOINTS_DIR} onto {DEVICE}...")

        self.tokenizer = AutoTokenizer.from_pretrained(target_source, **cache_kwargs)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            target_source,
            dtype=torch.bfloat16,
            **cache_kwargs
        ).to(DEVICE)

        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

    def _get_text_embeddings(self, batch_texts: list[str], max_length: int) -> torch.Tensor:
        tokens = self.tokenizer(
            batch_texts, 
            padding=True, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length
        ).to(DEVICE)
        
        with torch.no_grad():
            return self.llm.get_input_embeddings()(tokens.input_ids)


if __name__ == "__main__":
    trainer = QwenAlignmentTrainer(resume=True)
    trainer.train()