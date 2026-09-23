# PyTorch Dataset for Stage 3: cached FastConformer features paired with Mimi codes
import torch
from torch.utils.data import Dataset
import os
from concurrent.futures import ThreadPoolExecutor
from src.config import PROCESSED_TENSORS_DIR


class Stage3AudioCodeDataset(Dataset):
    """
    Custom PyTorch Dataset that loads cached FastConformer tensors and their
    corresponding Mimi codes directly in RAM for Stage 3 training. Reuses the
    same manifest.pt produced by extract_features.py + extract_mimi_codes.py,
    keeping only entries that have both a "features_path" and a "codes_path".
    """
    def __init__(self, manifest_filename: str = "manifest.pt"):
        self.manifest_path = os.path.join(PROCESSED_TENSORS_DIR, manifest_filename)

        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {self.manifest_path}. Please run feature extraction first.")

        full_index = torch.load(self.manifest_path)
        self.data_index = [item for item in full_index if "codes_path" in item]
        if not self.data_index:
            raise RuntimeError(
                f"No entries in {self.manifest_path} have Mimi codes yet. "
                "Run src/dataset/extract_mimi_codes.py first."
            )
        print(f"[*] Loaded PyTorch Dataset with {len(self.data_index)} audio-code pairs "
              f"(of {len(full_index)} total manifest entries)")

        # Eagerly cache all feature/code tensors in RAM so repeated epochs don't
        # re-hit the disk for every item. Loaded in parallel with a thread
        # pool since torch.load is I/O-bound and releases the GIL.
        print(f"[*] Caching {len(self.data_index)} feature/code tensors in RAM...")
        max_workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            features = list(executor.map(
                lambda item: torch.load(item["features_path"]), self.data_index
            ))
            codes = list(executor.map(
                lambda item: torch.load(item["codes_path"]), self.data_index
            ))
        self._cache = list(zip(features, codes))
        print("[*] Feature/code cache ready.")

    def __len__(self):
        """Return the total number of items in the dataset."""
        return len(self.data_index)

    def __getitem__(self, idx):
        """Fetches a single (audio_features, mimi_codes) pair from the in-RAM cache."""
        acoustic_features, mimi_codes = self._cache[idx]
        return acoustic_features, mimi_codes


# Quick local test to verify the loader. Batches use variable-length audio/codes, so real
# training goes through Stage3Collator (src/training/stage3_finetuning.py) rather than the
# default collate_fn used here for a plain single-sample sanity check.
if __name__ == "__main__":
    dataset = Stage3AudioCodeDataset()
    features, codes = dataset[0]

    print("\n--- TEST SAMPLE LOADED ---")
    print(f"Dataset size:  {len(dataset)}")
    print(f"Feature shape: {features.shape}")
    print(f"Codes shape:   {codes.shape}")
    print("Success: The Stage 3 dataset is ready for training!")
