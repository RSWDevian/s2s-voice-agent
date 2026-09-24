# Train LoRA adapers on the LLM backbone
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import LengthGroupedSampler
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from abc import ABC, abstractmethod

# Ensuring root directory is available for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import CHECKPOINTS_DIR, DEVICE, LLM_ID, LLM_DIM, TRAINED_MODELS_DIR

try:
    from src.dataset.dataset import FastConformerDataset
except ModuleNotFoundError:
    from src.dataset.dataset import FastConformerDataset

from src.models.projection import AudioToTextProjection

# Global Configuration for Stage 2
# Batch size reduced and learning rate lowered for LLM LoRA fine-tuning
# Tuning: if this OOMs on MPS, lower BATCH_SIZE further rather than adding
# gradient accumulation. Qwen's ~152K-token vocab makes the cross-entropy
# logits tensor (batch x seq_len x vocab_size) the main memory driver.
# NOTE: with length-grouped batching (see LengthGroupedSampler below), every
# batch in the first several steps is made of this dataset's longest audio
# sequences (~500 frames) padded together with no dilution from shorter
# samples — this is the actual worst-case memory batch, encountered
# deterministically on step 0 instead of randomly mid-epoch. Measured on
# this dataset: batch_size=4 OOMs on that first batch; batch_size=3 survives
# it with room to spare (~3.3GB steady-state, confirmed flat over 8 steps).
BATCH_SIZE = 3
EPOCHS = 3
LEARNING_RATE = 5e-5
LOG_EVERY_N_STEPS = 20  # how often to sync loss to CPU for the progress bar
MAX_GRAD_NORM = 1.0
MAX_CONSECUTIVE_BAD_STEPS = 5  # abort after this many non-finite steps in a row instead of training on garbage

# --- Backward-pass depth reduction ---
# LoRA adapters normally sit in every one of Qwen's 24 attention blocks, and
# the projector is trained jointly, so gradients must backprop through the
# full 24-layer stack every step (the projector sits at the very front of
# the sequence and needs a gradient). Restricting LoRA to only the top
# LORA_LAYERS_TO_TRANSFORM layers *and* freezing the projector lets
# autograd skip building/backpropagating through the lower (24-K) layers
# entirely, since nothing below the first adapted layer requires a
# gradient. Both must be changed together: leaving the projector trainable
# while restricting LoRA depth still forces a full-depth backward pass,
# since autograd must reach all the way back to the projector's weights
# regardless of where the LoRA adapters sit above it.
# Trade-off: the projector no longer gets refined by Stage 2's
# cross-entropy loss on top of Stage 1's MSE alignment, and fewer trainable
# LoRA layers means less adaptation capacity — validate convergence
# empirically and raise EPOCHS/LEARNING_RATE or widen the layer range if
# quality regresses. Set FREEZE_PROJECTOR_STAGE2 = False to fully revert to
# the previous (slower, jointly-trained) behavior.
FREEZE_PROJECTOR_STAGE2 = True
LORA_LAYERS_TO_TRANSFORM = list(range(16, 24))  # top 8 of Qwen2.5-0.5B's 24 layers

def backward_and_step(loss: torch.Tensor, optimizer, params) -> bool:
    """Backward + clipped optimizer step. Returns False, leaving the weights untouched, when the gradients are
    non-finite: one NaN step would otherwise poison AdamW's state and every weight from then on."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        return False
    optimizer.step()
    return True


class IndexedDataset(Dataset):
    """Wraps FastConformerDataset to also yield the sample index, so the
    collate function can look up that sample's precomputed text cache entry."""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        audio, _ = self.base_dataset[idx]
        return audio, idx


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
        resume: bool = False,
        max_steps_per_epoch: int | None = None,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        # Caps iterations per epoch for quick timing/smoke runs without
        # touching batch size or the dataset itself. None = full epoch.
        self.max_steps_per_epoch = max_steps_per_epoch

        # 1. Load Dataset
        try:
            self.dataset = FastConformerDataset()
        except FileNotFoundError as e:
            print(f"[!] Dataset not found. Ensure manifest.pt exists.")
            raise e

        # Detect encoder dimension dynamically
        sample_audio, _ = self.dataset[0]
        encoder_dim = sample_audio.shape[-1]

        # 2. Initialize the Trainable MLP Projector
        print("[*] Initializing the MLP projector...")
        self.projector = AudioToTextProjection(encoder_dim=encoder_dim, llm_dim=LLM_DIM).to(DEVICE)
        
        # --- WEIGHT LOADING LOGIC (Stage 1 vs Resume Stage 2) ---
        self.save_dir = os.path.join(TRAINED_MODELS_DIR, "stage2_adaptation")
        self.stage2_proj_path = os.path.join(self.save_dir, f"mlp_stage2_{self.__class__.__name__}.pth")
        self.stage2_lora_path = os.path.join(self.save_dir, f"lora_{self.__class__.__name__}")

        stage1_proj_path = os.path.join(TRAINED_MODELS_DIR, "trained_projector", f"mlp_stage1_QwenAlignmentTrainer.pth")

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

        # Freezing the projector here (rather than leaving it jointly
        # trainable) is what lets autograd skip the lower transformer layers
        # in the backward pass below — see LORA_LAYERS_TO_TRANSFORM.
        if FREEZE_PROJECTOR_STAGE2:
            self.projector.freeze_weights()
        else:
            self.projector.train()

        # 3. Load the specific LLM & Apply LoRA (Implemented by subclass)
        self._load_llm()

        # Match the projector's dtype to the LLM once here instead of casting
        # every batch in the training loop. Must happen after weight loading
        # (checkpoints are float32) and before the optimizer is built.
        self.projector.to(self.llm.dtype)

        # Text tokens/embeddings are a deterministic function of the fixed
        # transcripts and the frozen embedding table (embed_tokens is never
        # a LoRA target), so precompute them once instead of every step.
        self.text_cache = self._build_text_cache()

        # Build the DataLoader now that the text cache exists (the collate
        # function needs it) and the tokenizer's pad_token_id is known.
        # Length-grouped batching keeps padding (and therefore per-step
        # tensor shapes) far more consistent than random shuffling.
        audio_lengths = [audio.shape[0] for audio, _ in self.dataset]
        sampler = LengthGroupedSampler(batch_size=self.batch_size, lengths=audio_lengths)
        self.dataloader = DataLoader(
            IndexedDataset(self.dataset),
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=self._collate_fn,
        )

        # 4. Setup Optimizer (Projector params only included when not frozen)
        projector_params = [] if FREEZE_PROJECTOR_STAGE2 else list(self.projector.parameters())
        self.trainable_params = projector_params + [p for p in self.llm.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.trainable_params, lr=self.learning_rate)

    @abstractmethod
    def _load_llm(self):
        """Loads the LLM, tokenizers, and injects PEFT/LoRA adapters."""
        pass

    @abstractmethod
    def _get_text_tensors(self, batch_texts: list[str]) -> tuple:
        """Returns (text_embeds, input_ids, attention_mask)."""
        pass

    def _build_text_cache(self, chunk_size: int = 32) -> list:
        """One-time precompute of (unpadded text_embeds, unpadded input_ids)
        per dataset sample, keyed by dataset index."""
        print("[*] Precomputing text tokens + embeddings (one-time, frozen embedding table)...")
        all_texts = [text for _, text in self.dataset]
        cache = []
        for i in range(0, len(all_texts), chunk_size):
            chunk = all_texts[i:i + chunk_size]
            embeds, input_ids, attention_mask = self._get_text_tensors(chunk)
            for b in range(len(chunk)):
                mask = attention_mask[b].bool()  # works regardless of tokenizer padding_side
                cache.append((embeds[b][mask].detach().cpu(), input_ids[b][mask].detach().cpu()))
        print("[*] Text cache ready.")
        return cache

    def _collate_fn(self, batch):
        """Pads audio for this batch and re-pads each sample's cached
        (unpadded) text tokens/embeddings back into a batch tensor."""
        audio_tensors, idxs = zip(*batch)
        padded_audio = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)

        text_embeds_list = [self.text_cache[i][0] for i in idxs]
        text_ids_list = [self.text_cache[i][1] for i in idxs]
        lengths = torch.tensor([t.shape[0] for t in text_ids_list])

        text_embeds = pad_sequence(text_embeds_list, batch_first=True, padding_value=0.0)
        text_input_ids = pad_sequence(text_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        max_len = text_input_ids.shape[1]
        text_attention_mask = (torch.arange(max_len)[None, :] < lengths[:, None]).long()

        return padded_audio, text_embeds, text_input_ids, text_attention_mask

    def train(self):
        """Universal Stage 2 training loop using Cross-Entropy Loss."""
        print("\n[*] Commencing Stage 2 Adaptation Training Loop...")
        steps_per_epoch = len(self.dataloader)
        if self.max_steps_per_epoch is not None:
            steps_per_epoch = min(steps_per_epoch, self.max_steps_per_epoch)

        bad_steps = 0
        for epoch in range(self.epochs):
            total_loss_tensor = torch.zeros((), device=DEVICE)
            steps_run = 0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", total=steps_per_epoch)

            for step, (batch_audio, text_embeds, text_input_ids, text_attention_mask) in enumerate(progress_bar):
                if self.max_steps_per_epoch is not None and step >= self.max_steps_per_epoch:
                    break
                # Cached features are float32; cast to the projector/LLM's
                # dtype here so the projector's matmul isn't fed a
                # float32/bfloat16 mix (MPS aborts on that).
                batch_audio = batch_audio.to(DEVICE, dtype=self.llm.dtype)

                # Text tokens/embeddings come pre-tokenized and pre-embedded
                # from the text cache (built once in __init__) via _collate_fn;
                # only need to move them onto the accelerator here.
                text_embeds = text_embeds.to(DEVICE, dtype=self.llm.dtype)
                text_input_ids = text_input_ids.to(DEVICE)
                text_attention_mask = text_attention_mask.to(DEVICE)

                # --- FORWARD PASS: MODALITY 1 (AUDIO) ---
                # Projector and LLM share a dtype (set once in __init__), so no
                # per-step dtype casting is needed here.
                projected_audio = self.projector(batch_audio)
                audio_seq_len = projected_audio.shape[1]

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
                if not backward_and_step(loss, self.optimizer, self.trainable_params):
                    bad_steps += 1
                    print(f"\n[!] Non-finite gradients at epoch {epoch+1} step {step+1}; step skipped ({bad_steps} in a row).")
                    if bad_steps >= MAX_CONSECUTIVE_BAD_STEPS:
                        raise RuntimeError("Training diverged: repeated non-finite gradients. Nothing was saved from these steps.")
                    continue
                bad_steps = 0

                # Keep the running total on-device; only sync to CPU
                # periodically to avoid forcing an MPS queue drain every step.
                total_loss_tensor += loss.detach()
                steps_run += 1
                if (step + 1) % LOG_EVERY_N_STEPS == 0 or (step + 1) == steps_per_epoch:
                    progress_bar.set_postfix(loss=loss.item())
                    # Audio/text are padded per-batch, so every step's tensor
                    # shapes differ. MPS's caching allocator doesn't reliably
                    # reuse cached blocks across shapes, so its reserved pool
                    # grows unboundedly without this — periodically (not every
                    # step, since this forces a sync) hand unused blocks back.
                    if DEVICE.type == "mps":
                        torch.mps.empty_cache()

            avg_loss = (total_loss_tensor / max(steps_run, 1)).item()
            print(f"\n[+] Epoch {epoch+1} Completed. Average Loss: {avg_loss:.4f}")

            # Auto-save after every epoch
            self.save_checkpoint()

    def save_checkpoint(self):
        bad = [n for n, p in self.llm.named_parameters() if p.requires_grad and not torch.isfinite(p).all()]
        if bad:
            raise RuntimeError(f"Refusing to save a checkpoint with NaN/Inf weights (e.g. {bad[0]}).")
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
    def __init__(self, llm_id: str = LLM_ID, **kwargs):
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
            attn_implementation="sdpa",
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
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                layers_to_transform=LORA_LAYERS_TO_TRANSFORM,
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