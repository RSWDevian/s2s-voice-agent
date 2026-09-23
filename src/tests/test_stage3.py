# Tests for src/training/stage3_finetuning.py using a tiny random Qwen2 (no downloads, CPU, fp32)
import os

import pytest
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from src.config import CHECKPOINTS_DIR
from src.training import stage3_finetuning as s3

BASE_VOCAB = 100
VOCAB = s3.AudioVocab(base_id=BASE_VOCAB, num_codebooks=2, codebook_size=8)  # 18 new tokens


def tiny_llm(seed: int = 0):
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=BASE_VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(config)


def tiny_peft(seed: int = 0, dropout: float = 0.0):
    llm = tiny_llm(seed)
    s3.extend_llm_vocab(llm, VOCAB)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        trainable_token_indices=list(range(VOCAB.base_id, VOCAB.total_vocab_size)),
    )
    return get_peft_model(llm, config)


def make_batch(audio_lens, frame_counts, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    samples = [
        (torch.randn(a, 32, generator=generator),
         torch.randint(0, VOCAB.codebook_size, (VOCAB.num_codebooks, f), generator=generator))
        for a, f in zip(audio_lens, frame_counts)
    ]
    return s3.Stage3Collator(VOCAB, max_frames=50)(samples)


def batch_loss(model, batch):
    audio, audio_mask, target_ids = batch
    projected = audio  # stands in for the projector output (already llm_dim = 32)
    return s3.audio_token_loss(model, projected, audio_mask, target_ids, VOCAB)


def test_flatten_is_frame_major_and_round_trips():
    codes = torch.randint(0, 8, (2, 5))
    ids, complete = s3.flatten_codes(codes, VOCAB)
    assert complete and ids.shape == (10,)
    assert ids[0] == VOCAB.base_id + codes[0, 0]
    assert ids[1] == VOCAB.base_id + 8 + codes[1, 0]  # second codebook is offset by codebook_size
    assert torch.equal(s3.unflatten_ids(ids, VOCAB), codes)


def test_flatten_truncates_at_frame_boundary_and_rejects_bad_codes():
    ids, complete = s3.flatten_codes(torch.randint(0, 8, (2, 5)), VOCAB, max_frames=3)
    assert not complete and ids.shape == (6,)
    with pytest.raises(ValueError):
        s3.flatten_codes(torch.full((2, 3), 8), VOCAB)
    with pytest.raises(ValueError):
        s3.flatten_codes(torch.zeros(1, 3, dtype=torch.long), VOCAB)


def test_collator_left_pads_audio_and_marks_targets():
    audio, mask, targets = make_batch([5, 3], [4, 2])
    assert mask[1].tolist() == [1, 1, 1, 0, 0][::-1] and mask[0].sum() == 5
    assert torch.equal(audio[1, :2], torch.zeros(2, 32))
    assert targets[0, -1] == VOCAB.end_id and targets[1, 2 * 2] == VOCAB.end_id
    assert (targets[1, 2 * 2 + 1:] == s3.IGNORE_INDEX).all()

    long_codes = torch.randint(0, 8, (2, 60))
    _, _, truncated = s3.Stage3Collator(VOCAB, max_frames=50)([(torch.randn(3, 32), long_codes)])
    assert VOCAB.end_id not in truncated  # truncated clips must not teach an early stop


def test_only_lora_and_new_token_rows_train_and_old_rows_stay_frozen():
    model = tiny_peft()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n or "trainable_tokens" in n for n in trainable)

    embed = model.base_model.model.get_input_embeddings()
    old_ids, new_ids = torch.arange(VOCAB.base_id), torch.arange(VOCAB.base_id, VOCAB.total_vocab_size)
    old_before, new_before = embed(old_ids).detach().clone(), embed(new_ids).detach().clone()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    batch = make_batch([6, 4], [3, 2])
    for _ in range(3):
        loss = batch_loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert torch.equal(embed(old_ids).detach(), old_before)
    assert not torch.equal(embed(new_ids).detach(), new_before)


def test_loss_falls_when_overfitting_one_batch():
    model = tiny_peft()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    batch = make_batch([6, 4], [3, 2])
    first = batch_loss(model, batch).item()
    for _ in range(80):
        loss = batch_loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5 * first


def test_padding_does_not_change_the_loss():
    model = tiny_peft().eval()
    long_sample, short_sample = make_batch([8], [4], seed=1), make_batch([3], [2], seed=2)
    with torch.no_grad():
        loss_long, loss_short = batch_loss(model, long_sample), batch_loss(model, short_sample)
        n_long, n_short = (long_sample[2] != s3.IGNORE_INDEX).sum(), (short_sample[2] != s3.IGNORE_INDEX).sum()

        padded =s3.Stage3Collator(VOCAB, max_frames=50)([
            (long_sample[0][0], s3.unflatten_ids(long_sample[2][0][:-1], VOCAB)),
            (short_sample[0][0], s3.unflatten_ids(short_sample[2][0][:-1], VOCAB)),
        ])
        loss_padded = batch_loss(model, padded)
    expected = (n_long * loss_long + n_short * loss_short) / (n_long + n_short)
    assert torch.allclose(loss_padded, expected, atol=1e-4)


def test_saved_adapter_reloads_to_identical_logits(tmp_path):
    model = tiny_peft(seed=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    batch = make_batch([6, 4], [3, 2])
    for _ in range(3):
        loss = batch_loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    s3.save_stage3_adapter(model, VOCAB, str(tmp_path))

    fresh = tiny_llm(seed=0)  # same seed -> same base weights
    vocab = s3.AudioVocab.load(str(tmp_path))
    s3.extend_llm_vocab(fresh, vocab, seed=123)  # different init: the adapter must overwrite the new rows
    reloaded = PeftModel.from_pretrained(fresh, tmp_path)

    ids = torch.randint(0, VOCAB.total_vocab_size, (2, 7))
    with torch.no_grad():
        assert torch.allclose(model.eval()(input_ids=ids).logits, reloaded.eval()(input_ids=ids).logits, atol=1e-5)


def test_stage2_adapter_is_loaded_as_starting_point(tmp_path):
    stage2 = get_peft_model(tiny_llm(seed=0), LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], layers_to_transform=[2, 3],
    ))
    with torch.no_grad():
        for name, param in stage2.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.1)
    stage2.save_pretrained(tmp_path)

    llm = tiny_llm(seed=0)
    s3.extend_llm_vocab(llm, VOCAB)
    model = s3.build_peft_model(llm, VOCAB, str(tmp_path))

    saved = dict(stage2.named_parameters())
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert torch.equal(param, saved[name]), name
    assert model.peft_config["default"].layers_to_transform == [2, 3]


def test_nan_stage2_adapter_is_rejected(tmp_path):
    stage2 = get_peft_model(tiny_llm(seed=0), LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
    ))
    with torch.no_grad():
        next(p for n, p in stage2.named_parameters() if "lora_A" in n).fill_(float("nan"))
    stage2.save_pretrained(tmp_path)

    llm = tiny_llm(seed=0)
    s3.extend_llm_vocab(llm, VOCAB)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        s3.build_peft_model(llm, VOCAB, str(tmp_path))


def test_non_finite_step_is_skipped_and_finite_step_updates():
    model = tiny_peft()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    batch = make_batch([6, 4], [3, 2])
    before = [p.detach().clone() for p in params]

    nan_loss = batch_loss(model, batch) * float("nan")
    assert s3.backward_and_step(nan_loss, optimizer, params) is False
    assert all(torch.equal(p, b) for p, b in zip(params, before))  # weights untouched by the bad step

    assert s3.backward_and_step(batch_loss(model, batch), optimizer, params) is True
    assert any(not torch.equal(p, b) for p, b in zip(params, before))
    assert all(torch.isfinite(p).all() for p in params)


def test_saving_non_finite_adapter_is_refused(tmp_path):
    model = tiny_peft()
    with torch.no_grad():
        next(p for n, p in model.named_parameters() if p.requires_grad).fill_(float("nan"))
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        s3.save_stage3_adapter(model, VOCAB, str(tmp_path))
    assert not (tmp_path / "adapter_model.safetensors").exists()


def test_sample_lengths_terminates_and_counts_target_tokens():
    class NoIndexError(torch.utils.data.Dataset):  # like a placeholder dataset that never raises IndexError
        def __len__(self):
            return 3

        def __getitem__(self, idx):
            return torch.zeros(10 + idx, 32), torch.zeros(2, 4 + idx, dtype=torch.long)

    assert s3.sample_lengths(NoIndexError(), VOCAB) == [10 + 4 * 2 + 1, 11 + 5 * 2 + 1, 12 + 6 * 2 + 1]
    with pytest.raises(IndexError):
        s3.DummyAudioTokenDataset(num_samples=2)[2]
    assert len(list(s3.DummyAudioTokenDataset(num_samples=4))) == 4


def test_device_is_never_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert s3.select_device().type in ("mps", "cpu")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(CHECKPOINTS_DIR, "qwen_backbone", "tokenizer.json")), reason="Qwen tokenizer not downloaded"
)
def test_register_audio_tokens_on_real_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(CHECKPOINTS_DIR, "qwen_backbone"))
    text_before = tokenizer("नमस्ते hello world")["input_ids"]
    base = len(tokenizer)

    vocab = s3.register_audio_tokens(tokenizer)
    assert vocab.base_id == base and len(tokenizer) == vocab.total_vocab_size
    assert tokenizer.convert_tokens_to_ids("<|mimi_1_5|>") == base + 2048 + 5
    assert tokenizer.convert_tokens_to_ids("<|audio_end|>") == vocab.end_id
    assert tokenizer("नमस्ते hello world")["input_ids"] == text_before
