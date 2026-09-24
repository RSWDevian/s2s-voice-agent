# Tests for src/models/backbone.py. Requires the real Qwen checkpoint, so these are
# skipped in environments without it downloaded (mirrors the skip pattern used in
# test_stage3.py / test_generation.py for other real-weight-dependent checks).
import os

import pytest
import torch

from src.config import CHECKPOINTS_DIR
from src.models.backbone import QwenBackbone

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CHECKPOINTS_DIR, "qwen_backbone")), reason="Qwen backbone weights not downloaded"
)


def test_backbone_loads_tokenizer_with_pad_token():
    backbone = QwenBackbone(device="cpu")
    assert backbone.tokenizer.pad_token is not None


def test_backbone_freeze_sets_requires_grad_false():
    backbone = QwenBackbone(device="cpu", freeze=True)
    assert all(not p.requires_grad for p in backbone.parameters())
    assert not backbone.training


def test_backbone_unfreeze_sets_requires_grad_true():
    backbone = QwenBackbone(device="cpu", freeze=True)
    backbone.unfreeze_weights()
    assert all(p.requires_grad for p in backbone.parameters())
    assert backbone.training
