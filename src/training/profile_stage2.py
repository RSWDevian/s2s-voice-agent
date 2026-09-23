# Standalone timing breakdown for Stage 2 adaptation training.
# Run: python -m src.training.profile_stage2
import os
import sys
import time
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import DEVICE
from src.training.stage2_adaption import QwenAdaptationTrainer

PROFILE_STEPS = 25


def _sync():
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()


class _StageTimer:
    """Accumulates elapsed wall-clock time per named stage across steps."""
    def __init__(self):
        self.totals = {}
        self.counts = {}
        self._start = None
        self._name = None

    def start(self, name):
        _sync()
        self._name = name
        self._start = time.perf_counter()

    def stop(self):
        _sync()
        elapsed = time.perf_counter() - self._start
        self.totals[self._name] = self.totals.get(self._name, 0.0) + elapsed
        self.counts[self._name] = self.counts.get(self._name, 0) + 1

    def report(self):
        n_steps = max(self.counts.values())
        print(f"\n[*] Per-stage timing averaged over {n_steps} steps:")
        total_ms = sum(self.totals.values())
        for name, total in self.totals.items():
            avg_ms = (total / self.counts[name]) * 1000
            pct = (total / total_ms) * 100 if total_ms else 0.0
            print(f"    {name:<22} {avg_ms:8.2f} ms/step  ({pct:5.1f}%)")
        print(f"    {'TOTAL':<22} {(total_ms / n_steps) * 1000:8.2f} ms/step")


def main():
    print(f"[*] Profiling Stage 2 for {PROFILE_STEPS} steps on {DEVICE}...")
    trainer = QwenAdaptationTrainer(resume=True)
    timer = _StageTimer()

    step = 0
    for batch_audio, text_embeds, text_input_ids, text_attention_mask in trainer.dataloader:
        if step >= PROFILE_STEPS:
            break

        timer.start("data_to_device")
        batch_audio = batch_audio.to(DEVICE, dtype=trainer.llm.dtype)
        text_embeds = text_embeds.to(DEVICE, dtype=trainer.llm.dtype)
        text_input_ids = text_input_ids.to(DEVICE)
        text_attention_mask = text_attention_mask.to(DEVICE)
        timer.stop()

        timer.start("projector_forward")
        projected_audio = trainer.projector(batch_audio)
        audio_seq_len = projected_audio.shape[1]
        timer.stop()

        timer.start("qwen_forward")
        combined_embeds = torch.cat([projected_audio, text_embeds], dim=1)
        text_labels = text_input_ids.masked_fill(text_attention_mask == 0, -100)
        transformer_out = trainer.llm.base_model.model.model(inputs_embeds=combined_embeds, use_cache=False)
        hidden_states = transformer_out.last_hidden_state
        timer.stop()

        timer.start("lm_head_and_loss")
        predict_hidden = hidden_states[:, audio_seq_len - 1:-1, :]
        lm_head = trainer.llm.base_model.model.get_output_embeddings()
        logits = lm_head(predict_hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            text_labels.reshape(-1),
            ignore_index=-100,
        )
        timer.stop()

        timer.start("backward")
        trainer.optimizer.zero_grad()
        loss.backward()
        timer.stop()

        timer.start("optimizer_step")
        trainer.optimizer.step()
        timer.stop()

        step += 1

    timer.report()


if __name__ == "__main__":
    main()
