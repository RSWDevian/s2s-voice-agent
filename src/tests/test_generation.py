# Tests for the autoregressive generation loop in src/training/stage3_finetuning.py
# using the same tiny random Qwen2 fixtures as test_stage3.py (no downloads, CPU, fp32).
import os

import pytest
import torch
import torch.nn.functional as F

from src.config import CHECKPOINTS_DIR
from src.training import stage3_finetuning as s3
from src.tests.test_stage3 import VOCAB, tiny_peft


def _raw_logits(inner, hidden, vocab):
    """Same slice as _audio_head_logits but without the <|audio_start|> mask, so cache-vs-no-cache
    comparisons aren't tripped up by comparing -inf to -inf (which is NaN under allclose)."""
    head_weight = inner.get_output_embeddings().weight[vocab.base_id: vocab.total_vocab_size]
    return F.linear(hidden[:, -1, :], head_weight.to(hidden.dtype)).float().squeeze(0)


def test_trim_to_codes_strips_end_id_and_partial_frame():
    codes = torch.randint(0, VOCAB.codebook_size, (VOCAB.num_codebooks, 5))
    ids, complete = s3.flatten_codes(codes, VOCAB)
    assert complete
    with_end = torch.cat([ids, torch.tensor([VOCAB.end_id])])
    assert torch.equal(s3.trim_to_codes(with_end, VOCAB), ids)

    partial = ids[:-1]  # one token short of a whole frame
    trimmed = s3.trim_to_codes(partial, VOCAB)
    assert trimmed.numel() % VOCAB.num_codebooks == 0
    assert torch.equal(trimmed, ids[: trimmed.numel()])


def test_audio_head_logits_masks_start_id():
    model = tiny_peft()
    model.eval()
    inner = model.base_model.model
    hidden = torch.randn(1, 3, 32)
    logits = s3._audio_head_logits(inner, hidden, VOCAB)
    assert logits[VOCAB.start_id - VOCAB.base_id] == float("-inf")


def test_generated_ids_within_audio_vocab_range():
    model = tiny_peft()
    model.eval()
    projected_audio = torch.randn(1, 4, 32)
    generator = torch.Generator().manual_seed(0)
    ids = s3.generate_audio_tokens(
        model, VOCAB, projected_audio, max_new_tokens=10, temperature=1.0, top_k=5, generator=generator
    )
    assert ids.numel() > 0
    assert bool(((ids >= VOCAB.base_id) & (ids < VOCAB.total_vocab_size)).all())


def test_generate_respects_max_new_tokens_cap():
    model = tiny_peft()
    model.eval()
    projected_audio = torch.randn(1, 4, 32)
    ids = s3.generate_audio_tokens(model, VOCAB, projected_audio, max_new_tokens=5)
    assert ids.numel() <= 5


def test_generate_rejects_batch_greater_than_one():
    model = tiny_peft()
    model.eval()
    projected_audio = torch.randn(2, 4, 32)
    with pytest.raises(ValueError):
        s3.generate_audio_tokens(model, VOCAB, projected_audio, max_new_tokens=5)


def test_generate_stops_early_when_end_id_is_forced(monkeypatch):
    model = tiny_peft()
    model.eval()
    projected_audio = torch.randn(1, 3, 32)

    forced = torch.full((VOCAB.size,), -100.0)
    forced[VOCAB.end_id - VOCAB.base_id] = 100.0
    monkeypatch.setattr(s3, "_audio_head_logits", lambda inner, hidden, vocab: forced.clone())

    ids = s3.generate_audio_tokens(model, VOCAB, projected_audio, max_new_tokens=50)
    assert ids.tolist() == [VOCAB.end_id]


def test_kv_cache_step_matches_full_recompute_after_priming():
    model = tiny_peft()
    model.eval()
    inner = model.base_model.model
    projected_audio = torch.randn(1, 5, 32)

    start = torch.full((1, 1), VOCAB.start_id, dtype=torch.long)
    start_embed = inner.get_input_embeddings()(start)
    inputs_embeds = torch.cat([projected_audio, start_embed], dim=1)
    seq_len = inputs_embeds.shape[1]
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

    with torch.no_grad():
        no_cache_out = inner.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, use_cache=False,
        )
        cached_out = inner.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=s3.DynamicCache(), use_cache=True,
        )

    logits_no_cache = _raw_logits(inner, no_cache_out.last_hidden_state, VOCAB)
    logits_cached = _raw_logits(inner, cached_out.last_hidden_state, VOCAB)
    assert torch.allclose(logits_no_cache, logits_cached, atol=1e-4)


def test_kv_cache_step_matches_full_recompute_after_one_generated_token():
    model = tiny_peft()
    model.eval()
    inner = model.base_model.model
    projected_audio = torch.randn(1, 5, 32)

    start = torch.full((1, 1), VOCAB.start_id, dtype=torch.long)
    start_embed = inner.get_input_embeddings()(start)
    primed_embeds = torch.cat([projected_audio, start_embed], dim=1)
    primed_len = primed_embeds.shape[1]
    primed_mask = torch.ones(1, primed_len, dtype=torch.long)
    primed_positions = (primed_mask.cumsum(-1) - 1).clamp(min=0)

    next_token = torch.tensor([[VOCAB.base_id]], dtype=torch.long)  # first code token id

    with torch.no_grad():
        cached_out = inner.model(
            inputs_embeds=primed_embeds, attention_mask=primed_mask,
            position_ids=primed_positions, past_key_values=s3.DynamicCache(), use_cache=True,
        )
        next_embed = inner.get_input_embeddings()(next_token)
        step_out = inner.model(
            inputs_embeds=next_embed,
            attention_mask=torch.ones(1, primed_len + 1, dtype=torch.long),
            position_ids=torch.full((1, 1), primed_len, dtype=torch.long),
            past_key_values=cached_out.past_key_values, use_cache=True,
        )

        full_embeds = torch.cat([primed_embeds, next_embed], dim=1)
        full_len = full_embeds.shape[1]
        full_mask = torch.ones(1, full_len, dtype=torch.long)
        full_positions = (full_mask.cumsum(-1) - 1).clamp(min=0)
        full_out = inner.model(
            inputs_embeds=full_embeds, attention_mask=full_mask,
            position_ids=full_positions, use_cache=False,
        )

    logits_stepped = _raw_logits(inner, step_out.last_hidden_state, VOCAB)
    logits_full = _raw_logits(inner, full_out.last_hidden_state, VOCAB)
    assert torch.allclose(logits_stepped, logits_full, atol=1e-4)


def test_generate_audio_tokens_composes_with_trim_and_unflatten():
    model = tiny_peft()
    model.eval()
    projected_audio = torch.randn(1, 4, 32)
    ids = s3.generate_audio_tokens(model, VOCAB, projected_audio, max_new_tokens=9)

    trimmed = s3.trim_to_codes(ids, VOCAB)
    assert trimmed.numel() % VOCAB.num_codebooks == 0

    codes = s3.unflatten_ids(trimmed, VOCAB)
    assert codes.shape[0] == VOCAB.num_codebooks


@pytest.mark.skipif(
    not os.path.exists(os.path.join(CHECKPOINTS_DIR, "mimi_codec")), reason="Mimi codec weights not downloaded"
)
def test_mimi_codec_decode_shape():
    from src.models.decoder import MimiCodec

    codec = MimiCodec()
    codes = torch.randint(0, 2048, (8, 10))
    waveform = codec.decode(codes)
    assert waveform.dim() == 1
    assert waveform.numel() > 0

    audio = torch.randn(24000)
    round_trip = codec.decode(codec.encode(audio))
    assert round_trip.dim() == 1
    assert round_trip.numel() > 0
