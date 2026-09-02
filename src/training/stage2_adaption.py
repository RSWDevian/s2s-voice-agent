# Train LoRA adapers on the LLM backbone
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
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

# Global Configuration for Stage 2
# Batch size reduced and learning rate lowered for LLM LoRA fine-tuning
# Tuning: if this OOMs on MPS, halve BATCH_SIZE (try 4-6) rather than adding
# gradient accumulation. Qwen's ~152K-token vocab makes the cross-entropy
# logits tensor (batch x seq_len x vocab_size) the main memory driver.
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 5e-5
LOG_EVERY_N_STEPS = 20  # how often to sync loss to CPU for the progress bar

def pad_audio_collate_fn(batch):
    """Pads variable-length audio tensors for batching."""
    audio_tensors, text_transcripts = zip(*batch)
    padded_audio = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)
    return padded_audio, list(text_transcripts)


class BaseAdaptationTrainer(ABC):
    """
    Abstract Base Class for Stage 2 End-to-End Adaptation.
    Handles dataset streaming, Stage 1 weight loading, Stage 2 resuming, and the Cross-Entropy training loop.
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

        # 1. Load Dataset
        try:
            self.dataset = FastConformerDataset()
            self.dataloader = DataLoader(
                self.dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                collate_fn=pad_audio_collate_fn
            )
        except FileNotFoundError as e:
            print(f"[!] Dataset not found. Ensure manifest.pt exists.")
            raise e

        # Detect encoder dimension dynamically
        sample_audio, _ = self.dataset[0]
        encoder_dim = sample_audio.shape[-1]

        # 2. Initialize the Trainable MLP Projector
        print("[*] Initializing the MLP projector...")
        self.projector = AudioToTextProjection(encoder_dim=encoder_dim, llm_dim=1536).to(DEVICE)
        
        # --- WEIGHT LOADING LOGIC (Stage 1 vs Resume Stage 2) ---
        self.save_dir = os.path.join(CHECKPOINTS_DIR, "stage2_adaptation")
        self.stage2_proj_path = os.path.join(self.save_dir, f"mlp_stage2_{self.__class__.__name__}.pth")
        self.stage2_lora_path = os.path.join(self.save_dir, f"lora_{self.__class__.__name__}")
        
        stage1_proj_path = os.path.join(CHECKPOINTS_DIR, "trained_projector", f"mlp_stage1_QwenAlignmentTrainer.pth")

        if resume and os.path.exists(self.stage2_proj_path):
            print(f"[*] Resuming Stage 2: Loading projector weights from {self.stage2_proj_path}...")
            self.projector.load_state_dict(torch.load(self.stage2_proj_path, map_location=DEVICE))
            self.resume_lora = True
        else:
            if not os.path.exists(stage1_proj_path):
                raise FileNotFoundError(f"[!] Stage 1 weights missing at {stage1_proj_path}. Run Stage 1 first.")
            print(f"[*] Starting Stage 2: Loading base projector weights from Stage 1...")
            self.projector.load_state_dict(torch.load(stage1_proj_path, map_location=DEVICE))
            self.resume_lora = False
        
        self.projector.train()

        # 3. Load the specific LLM & Apply LoRA (Implemented by subclass)
        self._load_llm()

        # Match the projector's dtype to the LLM once here instead of casting
        # every batch in the training loop. Must happen after weight loading
        # (checkpoints are float32) and before the optimizer is built.
        self.projector.to(self.llm.dtype)

        # 4. Setup Optimizer (Optimizing BOTH Projector and LoRA weights)
        trainable_params = list(self.projector.parameters()) + [p for p in self.llm.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

    @abstractmethod
    def _load_llm(self):
        """Loads the LLM, tokenizers, and injects PEFT/LoRA adapters."""
        pass

    @abstractmethod
    def _get_text_tensors(self, batch_texts: list[str]) -> tuple:
        """Returns (text_embeds, input_ids, attention_mask)."""
        pass

    def train(self):
        """Universal Stage 2 training loop using Cross-Entropy Loss."""
        print("\n[*] Commencing Stage 2 Adaptation Training Loop...")
        for epoch in range(self.epochs):
            total_loss_tensor = torch.zeros((), device=DEVICE)
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for step, (batch_audio, batch_texts) in enumerate(progress_bar):
                # Cached features are float32; cast to the projector/LLM's
                # dtype here so the projector's matmul isn't fed a
                # float32/bfloat16 mix (MPS aborts on that).
                batch_audio = batch_audio.to(DEVICE, dtype=self.llm.dtype)

                # --- FORWARD PASS: MODALITY 1 (AUDIO) ---
                # Projector and LLM share a dtype (set once in __init__), so no
                # per-step dtype casting is needed here.
                projected_audio = self.projector(batch_audio)
                audio_seq_len = projected_audio.shape[1]

                # --- FORWARD PASS: MODALITY 2 (TEXT) ---
                text_embeds, text_input_ids, text_attention_mask = self._get_text_tensors(batch_texts)

                # --- CONCATENATE MODALITIES ---
                # Audio acts as the prompt, Text acts as the generation target
                combined_embeds = torch.cat([projected_audio, text_embeds], dim=1)

                # Mask padding tokens in the text. Only text positions are ever
                # predicted, so no audio-side labels are needed.
                text_labels = text_input_ids.masked_fill(text_attention_mask == 0, -100)

                # --- LLM FORWARD: TRANSFORMER BODY ONLY (no lm_head yet) ---
                # Calling the inner transformer directly (bypassing the
                # CausalLM wrapper's forward) avoids projecting every
                # audio-prefix position through the ~152K-vocab lm_head only
                # to discard it via -100 masking. Passing labels= to the full
                # wrapper still computes logits for the *entire* sequence
                # internally regardless of the mask, so that overhead can only
                # be avoided by skipping lm_head ourselves on those positions.
                transformer_out = self.llm.base_model.model.model(inputs_embeds=combined_embeds, use_cache=False)
                hidden_states = transformer_out.last_hidden_state

                # --- LM HEAD + LOSS ON TEXT POSITIONS ONLY ---
                # Position (audio_seq_len - 1) is the last audio-prefix
                # position; its next-token prediction is the first text
                # token. This slice mirrors the standard causal-LM label
                # shift (logits[:-1] predicts labels[1:]) restricted to the
                # text region only.
                predict_hidden = hidden_states[:, audio_seq_len - 1:-1, :]
                lm_head = self.llm.base_model.model.get_output_embeddings()
                logits = lm_head(predict_hidden)
                # Upcast to float32 for the softmax/loss, matching what
                # transformers' built-in loss does internally for stability.
                loss = nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    text_labels.reshape(-1),
                    ignore_index=-100,
                )
                
                # --- BACKWARD PASS ---
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Keep the running total on-device; only sync to CPU
                # periodically to avoid forcing an MPS queue drain every step.
                total_loss_tensor += loss.detach()
                if (step + 1) % LOG_EVERY_N_STEPS == 0 or (step + 1) == len(self.dataloader):
                    progress_bar.set_postfix(loss=loss.item())
                    # Audio/text are padded per-batch, so every step's tensor
                    # shapes differ. MPS's caching allocator doesn't reliably
                    # reuse cached blocks across shapes, so its reserved pool
                    # grows unboundedly without this — periodically (not every
                    # step, since this forces a sync) hand unused blocks back.
                    if DEVICE.type == "mps":
                        torch.mps.empty_cache()

            avg_loss = (total_loss_tensor / len(self.dataloader)).item()
            print(f"\n[+] Epoch {epoch+1} Completed. Average Loss: {avg_loss:.4f}")

            # Auto-save after every epoch
            self.save_checkpoint()

    def save_checkpoint(self):
        os.makedirs(self.save_dir, exist_ok=True)
        # Save LoRA adapters
        self.llm.save_pretrained(self.stage2_lora_path)
        # Save refined projector weights
        torch.save(self.projector.state_dict(), self.stage2_proj_path)
        print(f"\n[+] Stage 2 Checkpoint safely saved to {self.save_dir}")


class QwenAdaptationTrainer(BaseAdaptationTrainer):
    """
    Stage 2 Implementation specifically tailored for the Qwen2.5 architecture.
    """
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

        if self.resume_lora and os.path.exists(self.stage2_lora_path):
            print(f"[*] Resuming Stage 2: Loading existing LoRA adapters from {self.stage2_lora_path}...")
            self.llm = PeftModel.from_pretrained(self.llm, self.stage2_lora_path, is_trainable=True)
        else:
            print("[*] Injecting fresh LoRA adapters into Qwen...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                r=8, 
                lora_alpha=16, 
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] 
            )
            self.llm = get_peft_model(self.llm, lora_config)
        
        self.llm.print_trainable_parameters()
        self.llm.to(DEVICE)

    def _get_text_tensors(self, batch_texts: list[str]) -> tuple:
        tokens = self.tokenizer(
            batch_texts, 
            padding=True, 
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            # Bypass PEFT wrapper to grab raw embeddings directly from the base model
            text_embeds = self.llm.base_model.model.get_input_embeddings()(tokens.input_ids)
            
        return text_embeds, tokens.input_ids, tokens.attention_mask


if __name__ == "__main__":
    # resume=True will automatically handle picking up where you left off
    trainer = QwenAdaptationTrainer(resume=True)
    trainer.train()