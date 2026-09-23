# Offline script: second pass over the Stage 1/2 manifest -> Mimi codes -> .pt
# Streams the same HF dataset, in the same order, as extract_features.py, and attaches a
# "codes_path" to each existing manifest entry so Stage 3 can index against the same ids.
import os
import sys
import librosa
import torch
from tqdm import tqdm
from datasets import load_dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import PROCESSED_TENSORS_DIR, DATASET_ID, MIMI_SAMPLE_RATE
from src.models.decoder import MimiCodec


def run_mimi_extraction(dataset_name: str = DATASET_ID):
    manifest_path = os.path.join(PROCESSED_TENSORS_DIR, "manifest.pt")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest file not found at {manifest_path}. Run extract_features.py first."
        )
    manifest = torch.load(manifest_path)

    # Resume logic: pick up right after the last entry that already has codes.
    start_index = sum(1 for item in manifest if "codes_path" in item)
    if start_index >= len(manifest):
        print(f"[*] All {len(manifest)} manifest entries already have Mimi codes. Nothing to do.")
        return
    print(f"[*] Resuming Mimi code extraction from index {start_index} of {len(manifest)} ...")

    print(f"[*] Streaming samples from Hugging Face for Mimi encoding ...")
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    if start_index > 0:
        dataset = dataset.skip(start_index)
    codec = MimiCodec()

    try:
        for i, items in enumerate(tqdm(dataset, desc="Encoding audio", initial=start_index), start=start_index):
            if i >= len(manifest):
                # Don't run past what extract_features.py has already indexed; run it again
                # first if more samples are needed.
                break

            raw_audio = items["audio"]["array"]
            sample_rate = items["audio"]["sampling_rate"]
            waveform = torch.tensor(raw_audio, dtype=torch.float32)
            if sample_rate != MIMI_SAMPLE_RATE:
                waveform = torch.from_numpy(
                    librosa.resample(waveform.numpy(), orig_sr=sample_rate, target_sr=MIMI_SAMPLE_RATE)
                )

            codes = codec.encode(waveform)
            codes_filename = f"{manifest[i]['id']}_codes.pt"
            codes_filepath = os.path.join(PROCESSED_TENSORS_DIR, codes_filename)
            torch.save(codes, codes_filepath)
            manifest[i]["codes_path"] = codes_filepath

            if (i + 1) % 100 == 0:
                torch.save(manifest, manifest_path)
    except KeyboardInterrupt:
        print(f"\n[!] Process paused by user (Ctrl + C)...")
    finally:
        torch.save(manifest, manifest_path)
        done = sum(1 for item in manifest if "codes_path" in item)
        print(f"[*] Extraction saved successfully")
        print(f"[*] Manifest entries with Mimi codes: {done}/{len(manifest)}")


if __name__ == "__main__":
    run_mimi_extraction()
