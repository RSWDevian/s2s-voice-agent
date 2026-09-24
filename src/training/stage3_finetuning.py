# Stage 3: teach the LoRA-adapted LLM to emit Mimi audio tokens (speech in -> audio tokens out)
import json
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import LengthGroupedSampler

# Ensuring root directory is available for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import CHECKPOINTS_DIR, LLM_DIM, LLM_ID
from src.models.projection import AudioToTextProjection

# Global Configuration for Stage 3 (untuned starting points)
BATCH_SIZE = 2
EPOCHS = 3
LORA_LR = 1e-4
TOKEN_LR = 5e-4  # freshly initialised audio-token rows need a larger step than the LoRA weights
MAX_GRAD_NORM = 1.0
MAX_CONSECUTIVE_BAD_STEPS = 5  # abort after this many non-finite steps in a row instead of training on garbage
LOG_EVERY_N_STEPS = 20

# Mimi layout: 12.5 frames/s, CODEBOOK_SIZE codes per codebook. Every frame becomes
# NUM_CODEBOOKS consecutive tokens, so the LLM sees NUM_CODEBOOKS * 12.5 tokens per second of audio.
NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 2048
MAX_TARGET_FRAMES = 100  # 8 s; longer targets are cut at a frame boundary (keeps memory bounded)

# Continue from the Stage 2 adapter (keeps its audio understanding) instead of a fresh LoRA.
INIT_FROM_STAGE2 = True
FRESH_LORA_LAYERS = None  # only used for a fresh LoRA; None = all layers (backward is full-depth anyway)

STAGE1_PROJECTOR_PATH = os.path.join(CHECKPOINTS_DIR, "trained_projector", "mlp_stage1_QwenAlignmentTrainer.pth")
STAGE2_LORA_PATH = os.path.join(CHECKPOINTS_DIR, "stage2_adaptation", "lora_QwenAdaptationTrainer")
STAGE3_LORA_PATH = os.path.join(CHECKPOINTS_DIR, "stage3_lora")

IGNORE_INDEX = -100
VOCAB_FILE = "audio_vocab.json"


def select_device() -> torch.device:
    # Deliberately never CUDA: this stage targets Apple Silicon unified memory.
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


@dataclass(frozen=True)
class AudioVocab:
    """Id layout of the tokens appended to the LLM vocabulary:
    [base_id, base_id + K*C) codes (codebook-major), then <|audio_start|>, then <|audio_end|>."""
    base_id: int
    num_codebooks: int = NUM_CODEBOOKS
    codebook_size: int = CODEBOOK_SIZE

    @property
    def num_code_tokens(self) -> int:
        return self.num_codebooks * self.codebook_size

    @property
    def start_id(self) -> int:
        return self.base_id + self.num_code_tokens

    @property
    def end_id(self) -> int:
        return self.start_id + 1

    @property
    def size(self) -> int:
        return self.num_code_tokens + 2

    @property
    def total_vocab_size(self) -> int:
        return self.base_id + self.size

    def token_names(self) -> list[str]:
        codes = [f"<|mimi_{k}_{c}|>" for k in range(self.num_codebooks) for c in range(self.codebook_size)]
        return codes + ["<|audio_start|>", "<|audio_end|>"]

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, VOCAB_FILE), "w") as f:
            json.dump(
                {"base_id": self.base_id, "num_codebooks": self.num_codebooks, "codebook_size": self.codebook_size},
                f, indent=2,
            )

    @classmethod
    def load(cls, directory: str) -> "AudioVocab":
        with open(os.path.join(directory, VOCAB_FILE)) as f:
            return cls(**json.load(f))


def register_audio_tokens(tokenizer, num_codebooks: int = NUM_CODEBOOKS, codebook_size: int = CODEBOOK_SIZE) -> AudioVocab:
    """Adds the Mimi code tokens plus start/end markers to the tokenizer and returns their id layout."""
    vocab = AudioVocab(base_id=len(tokenizer), num_codebooks=num_codebooks, codebook_size=codebook_size)
    names = vocab.token_names()
    tokenizer.add_tokens(names, special_tokens=True)
    ids = tokenizer.convert_tokens_to_ids(names)
    if ids != list(range(vocab.base_id, vocab.total_vocab_size)):
        raise RuntimeError("Audio tokens did not receive contiguous ids; tokenizer may already contain them.")
    return vocab


def extend_llm_vocab(llm, vocab: AudioVocab, seed: int = 0):
    """Grows embed_tokens/lm_head to fit the audio tokens and initialises the new rows near the
    old embedding distribution (mean + noise) so they start distinct but on-scale."""
    if not llm.config.tie_word_embeddings:
        raise NotImplementedError("Only tied embed_tokens/lm_head is supported (trainable token rows cover both).")

    llm.resize_token_embeddings(vocab.total_vocab_size, mean_resizing=False)
    weight = llm.get_input_embeddings().weight
    with torch.no_grad():
        old = weight[: vocab.base_id].detach().float().cpu()
        generator = torch.Generator().manual_seed(seed)
        new_rows = old.mean(0) + torch.randn(vocab.size, old.shape[1], generator=generator) * old.std()
        weight[vocab.base_id: vocab.total_vocab_size] = new_rows.to(weight.device, weight.dtype)


def flatten_codes(codes: torch.Tensor, vocab: AudioVocab, max_frames: int = MAX_TARGET_FRAMES):
    """[K', T'] Mimi codes -> frame-major token ids [T'*K] with per-codebook offsets.
    Returns (ids, complete); complete is False when the clip was truncated to max_frames."""
    if codes.dim() != 2 or codes.shape[0] < vocab.num_codebooks:
        raise ValueError(f"Expected Mimi codes of shape [>= {vocab.num_codebooks}, T], got {tuple(codes.shape)}")
    codes = codes[: vocab.num_codebooks].long()
    if codes.min() < 0 or codes.max() >= vocab.codebook_size:
        raise ValueError(f"Mimi codes must be in [0, {vocab.codebook_size})")

    complete = codes.shape[1] <= max_frames
    codes = codes[:, :max_frames]
    offsets = (torch.arange(vocab.num_codebooks) * vocab.codebook_size + vocab.base_id).unsqueeze(1)
    return (codes + offsets).t().reshape(-1), complete


def unflatten_ids(ids: torch.Tensor, vocab: AudioVocab) -> torch.Tensor:
    """Inverse of flatten_codes for whole frames: token ids [T*K] -> Mimi codes [K, T]."""
    frames = ids.reshape(-1, vocab.num_codebooks).t()
    offsets = (torch.arange(vocab.num_codebooks) * vocab.codebook_size + vocab.base_id).unsqueeze(1)
    return frames - offsets


class Stage3Collator:
    """Left-pads audio (so real audio ends right before <|audio_start|>) and right-pads target ids."""
    def __init__(self, vocab: AudioVocab, max_frames: int = MAX_TARGET_FRAMES):
        self.vocab = vocab
        self.max_frames = max_frames

    def __call__(self, batch):
        audios, code_list = zip(*batch)
        lengths = [a.shape[0] for a in audios]
        max_len = max(lengths)
        padded_audio = audios[0].new_zeros(len(audios), max_len, audios[0].shape[-1])
        audio_mask = torch.zeros(len(audios), max_len, dtype=torch.long)
        for i, (audio, n) in enumerate(zip(audios, lengths)):
            padded_audio[i, max_len - n:] = audio
            audio_mask[i, max_len - n:] = 1

        targets = []
        for codes in code_list:
            ids, complete = flatten_codes(codes, self.vocab, self.max_frames)
            if complete:  # a truncated clip must not teach the model to stop early
                ids = torch.cat([ids, torch.tensor([self.vocab.end_id])])
            targets.append(ids)
        target_ids = pad_sequence(targets, batch_first=True, padding_value=IGNORE_INDEX)
        return padded_audio, audio_mask, target_ids


def audio_token_loss(llm, projected_audio, audio_mask, target_ids, vocab: AudioVocab) -> torch.Tensor:
    """Teacher-forced next-audio-token cross-entropy.
    Sequence: [audio][<|audio_start|>][target[:-1]]; the hidden state at <|audio_start|> predicts target[0]."""
    inner = llm.base_model.model
    batch, audio_len, _ = projected_audio.shape
    valid = target_ids != IGNORE_INDEX

    start = torch.full((batch, 1), vocab.start_id, dtype=torch.long, device=target_ids.device)
    input_ids = torch.cat([start, target_ids[:, :-1].clamp(min=0)], dim=1)
    token_embeds = inner.get_input_embeddings()(input_ids).to(projected_audio.dtype)
    inputs_embeds = torch.cat([projected_audio, token_embeds], dim=1)

    token_mask = torch.cat([torch.ones_like(start), valid[:, :-1].long()], dim=1)
    attention_mask = torch.cat([audio_mask, token_mask], dim=1)
    # Left padding shifts absolute positions, so derive them from the mask instead of arange.
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

    hidden = inner.model(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
    ).last_hidden_state
    predict_hidden = hidden[:, audio_len:, :]

    # Targets are always audio tokens, so score only that slice of the vocabulary (~16K of ~168K columns).
    # .weight on the PEFT-wrapped tied head includes the trainable rows, so gradients reach them.
    head_weight = inner.get_output_embeddings().weight[vocab.base_id: vocab.total_vocab_size]
    logits = F.linear(predict_hidden, head_weight.to(predict_hidden.dtype))
    labels = (target_ids - vocab.base_id).masked_fill(~valid, IGNORE_INDEX)
    return F.cross_entropy(logits.float().reshape(-1, vocab.size), labels.reshape(-1), ignore_index=IGNORE_INDEX)


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


def load_backbone(device: torch.device):
    """Loads the Qwen tokenizer and bfloat16 LLM (local folder first, HF cache otherwise), as Stage 2 does."""
    local_backbone = os.path.join(CHECKPOINTS_DIR, "qwen_backbone")
    if os.path.exists(local_backbone) and len(os.listdir(local_backbone)) > 0:
        source, cache_kwargs = local_backbone, {}
    else:
        source, cache_kwargs = LLM_ID, {"cache_dir": CHECKPOINTS_DIR}
    print(f"[*] Loading Qwen backbone from {source} onto {device}...")

    tokenizer = AutoTokenizer.from_pretrained(source, **cache_kwargs)
    llm = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation="sdpa", **cache_kwargs
    ).to(device)
    return tokenizer, llm


def _load_lora_weights(model, saved: dict):
    """Copies saved LoRA tensors into the model. PEFT's own loader can't be used here: it also demands
    the trainable-token weights, which a Stage 2 adapter (saved without them) does not contain."""
    params = dict(model.named_parameters())
    loaded = set()
    with torch.no_grad():
        for key, value in saved.items():
            name = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
            if name not in params:
                raise RuntimeError(f"Stage 2 adapter key has no matching parameter: {key}")
            if not torch.isfinite(value).all():
                raise RuntimeError(
                    f"Stage 2 adapter tensor {key} contains NaN/Inf (the Stage 2 run diverged). "
                    "Retrain Stage 2, or set INIT_FROM_STAGE2 = False to start a fresh LoRA."
                )
            params[name].copy_(value)
            loaded.add(name)
    missing = [n for n in params if "lora_" in n and n not in loaded]
    if missing:
        raise RuntimeError(f"Stage 2 adapter is missing weights for: {missing[:5]}")


def build_peft_model(llm, vocab: AudioVocab, stage2_path: str | None = None):
    """LoRA (q/k/v/o) plus trainable rows for only the new audio tokens; all base weights stay frozen.
    With a Stage 2 adapter available, reuses its LoRA config and weights as the starting point."""
    token_ids = list(range(vocab.base_id, vocab.total_vocab_size))
    stage2_weights = os.path.join(stage2_path, "adapter_model.safetensors") if stage2_path else None

    if stage2_weights and os.path.exists(stage2_weights):
        print(f"[*] Continuing from Stage 2 adapter at {stage2_path}...")
        config = LoraConfig.from_pretrained(stage2_path)
        config.trainable_token_indices = token_ids
        config.inference_mode = False
        model = get_peft_model(llm, config)
        _load_lora_weights(model, load_file(stage2_weights))
    else:
        print("[*] Injecting fresh LoRA adapters...")
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            layers_to_transform=FRESH_LORA_LAYERS,
            trainable_token_indices=token_ids,
        )
        model = get_peft_model(llm, config)

    model.print_trainable_parameters()
    return model


def save_stage3_adapter(model, vocab: AudioVocab, directory: str):
    """Adapter files only (adapter_config.json + adapter_model.safetensors, which already carry the trained
    audio-token rows) plus the small id-layout file needed to rebuild the vocabulary on load.
    save_embedding_layers=False stops PEFT from also dumping the whole resized embedding matrix."""
    bad = [n for n, p in model.named_parameters() if p.requires_grad and not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(f"Refusing to save an adapter with NaN/Inf weights (e.g. {bad[0]}).")
    model.save_pretrained(directory, save_embedding_layers=False)
    vocab.save(directory)


def load_stage3_llm(adapter_dir: str = STAGE3_LORA_PATH, device: torch.device | None = None):
    """Rebuilds the Stage 3 model from a saved adapter directory (for inference / verification)."""
    device = device or select_device()
    vocab = AudioVocab.load(adapter_dir)
    _, llm = load_backbone(device)
    extend_llm_vocab(llm, vocab)
    return PeftModel.from_pretrained(llm, adapter_dir), vocab


def sample_lengths(dataset: Dataset, vocab: AudioVocab) -> list[int]:
    """Per-sample sequence length (audio frames + target tokens) for length-grouped batching.
    Indexes by range(len(dataset)) because map-style datasets need not raise IndexError to end `for x in dataset`."""
    lengths = []
    for i in range(len(dataset)):
        audio, codes = dataset[i]
        lengths.append(audio.shape[0] + min(codes.shape[1], MAX_TARGET_FRAMES) * vocab.num_codebooks + 1)
    return lengths


class DummyAudioTokenDataset(Dataset):
    """Placeholder yielding (FastConformer-style features [T, D], Mimi codes [K, T']); swap in the real dataset."""
    def __init__(self, num_samples: int = 32, encoder_dim: int = 512, seed: int = 0):
        self.num_samples = num_samples
        self.encoder_dim = encoder_dim
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not 0 <= idx < self.num_samples:
            raise IndexError(idx)
        generator = torch.Generator().manual_seed(self.seed + idx)
        audio_frames = int(torch.randint(40, 120, (1,), generator=generator))
        code_frames = int(torch.randint(10, 40, (1,), generator=generator))
        audio = torch.randn(audio_frames, self.encoder_dim, generator=generator)
        codes = torch.randint(0, CODEBOOK_SIZE, (NUM_CODEBOOKS, code_frames), generator=generator)
        return audio, codes


class Stage3Trainer:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = BATCH_SIZE,
        epochs: int = EPOCHS,
        lora_lr: float = LORA_LR,
        token_lr: float = TOKEN_LR,
        init_from_stage2: bool = INIT_FROM_STAGE2,
        max_steps_per_epoch: int | None = None,
        save_dir: str = STAGE3_LORA_PATH,
    ):
        self.device = select_device()
        self.epochs = epochs
        self.max_steps_per_epoch = max_steps_per_epoch
        self.save_dir = save_dir

        sample_audio, _ = dataset[0]
        encoder_dim = sample_audio.shape[-1]

        # Frozen Stage 1 projector, cast once to the LLM dtype.
        self.projector = AudioToTextProjection(
            encoder_dim=encoder_dim, llm_dim=LLM_DIM, weights_path=STAGE1_PROJECTOR_PATH,
            device=self.device, freeze=True,
        ).to(torch.bfloat16)

        tokenizer, llm = load_backbone(self.device)
        self.vocab = register_audio_tokens(tokenizer)
        extend_llm_vocab(llm, self.vocab)
        print(f"[*] Added {self.vocab.size} audio tokens (ids {self.vocab.base_id}..{self.vocab.total_vocab_size - 1}).")
        self.llm = build_peft_model(llm, self.vocab, STAGE2_LORA_PATH if init_from_stage2 else None)
        self.llm.train()

        lengths = sample_lengths(dataset, self.vocab)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=LengthGroupedSampler(batch_size=batch_size, lengths=lengths),
            collate_fn=Stage3Collator(self.vocab),
        )

        named = [(n, p) for n, p in self.llm.named_parameters() if p.requires_grad]
        lora_params = [p for n, p in named if "lora_" in n]
        token_params = [p for n, p in named if "trainable_tokens" in n]
        stray = [n for n, _ in named if "lora_" not in n and "trainable_tokens" not in n]
        if stray:
            raise RuntimeError(f"Unexpected trainable parameters (only LoRA + new token rows allowed): {stray[:5]}")
        self.trainable_params = lora_params + token_params
        self.optimizer = torch.optim.AdamW(
            [{"params": lora_params, "lr": lora_lr}, {"params": token_params, "lr": token_lr}]
        )

    def train(self):
        print("\n[*] Commencing Stage 3 Audio-Token Training Loop...")
        steps_per_epoch = len(self.dataloader)
        if self.max_steps_per_epoch is not None:
            steps_per_epoch = min(steps_per_epoch, self.max_steps_per_epoch)

        bad_steps = 0
        for epoch in range(self.epochs):
            total_loss = torch.zeros((), device=self.device)
            steps_run = 0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", total=steps_per_epoch)

            for step, (audio, audio_mask, target_ids) in enumerate(progress_bar):
                if step >= steps_per_epoch:
                    break
                audio = audio.to(self.device, dtype=torch.bfloat16)
                audio_mask = audio_mask.to(self.device)
                target_ids = target_ids.to(self.device)

                with torch.no_grad():
                    projected_audio = self.projector(audio)

                loss = audio_token_loss(self.llm, projected_audio, audio_mask, target_ids, self.vocab)

                if not backward_and_step(loss, self.optimizer, self.trainable_params):
                    bad_steps += 1
                    print(f"\n[!] Non-finite gradients at epoch {epoch+1} step {step+1}; step skipped ({bad_steps} in a row).")
                    if bad_steps >= MAX_CONSECUTIVE_BAD_STEPS:
                        raise RuntimeError("Training diverged: repeated non-finite gradients. Nothing was saved from these steps.")
                    continue
                bad_steps = 0

                # Keep the running total on-device; only sync to CPU periodically.
                total_loss += loss.detach()
                steps_run += 1

                # Every batch has a different padded audio/code length, and MPS's caching
                # allocator doesn't reliably reuse blocks across shapes, so its reserved pool
                # grows (and step time climbs) unboundedly without this. Cleared every step
                # (not just periodically) since Stage 3's shape variance is wide enough that
                # waiting for the periodic log tick let the reserved pool -- and step time --
                # grow noticeably in between.
                if self.device.type == "mps":
                    torch.mps.empty_cache()

                if (step + 1) % LOG_EVERY_N_STEPS == 0 or (step + 1) == steps_per_epoch:
                    progress_bar.set_postfix(loss=loss.item())

            print(f"\n[+] Epoch {epoch+1} Completed. Average Loss: {(total_loss / max(steps_run, 1)).item():.4f}")
            self.save_checkpoint()

    def save_checkpoint(self):
        save_stage3_adapter(self.llm, self.vocab, self.save_dir)
        print(f"[+] Stage 3 adapter saved to {self.save_dir}")


if __name__ == "__main__":
    from src.dataset.stage3_dataset import Stage3AudioCodeDataset

    trainer = Stage3Trainer(Stage3AudioCodeDataset())
    trainer.train()
