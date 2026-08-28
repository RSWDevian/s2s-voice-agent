# PyTorch Dataset for loading cached .pt tensors
import torch
from torch.utils.data import Dataset, DataLoader
import os
from src.config import PROCESSED_TENSORS_DIR

class FastConformerDataset(Dataset):
    """
    Custom PyTorch Dataset that loads cached FastConformer tensors and 
    their corresponding text transripts directly in RAM for training.
    """
    def __init__(self, manifest_filename: str = "manifest.pt"):
        self.manifest_path = os.path.join(PROCESSED_TENSORS_DIR, manifest_filename)

        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {self.manifest_path}. Please run feature extraction first.")

        self.data_index = torch.load(self.manifest_path)
        print(f"[*] Loaded PyTorch Dataset with {len(self.data_index)} audio text pairs")

        # Eagerly cache all feature tensors in RAM so repeated epochs don't
        # re-hit the disk for every item.
        print(f"[*] Caching {len(self.data_index)} feature tensors in RAM...")
        self._cache = [
            (torch.load(item["features_path"]), item["text_transcript"])
            for item in self.data_index
        ]
        print("[*] Feature cache ready.")

    def __len__(self):
        """Return the total number of items in the dataset."""
        return len(self.data_index)

    def __getitem__(self, idx):
        """Fetches a single data pair from the in-RAM cache."""
        acoustic_features, text_transcript = self._cache[idx]
        return acoustic_features, text_transcript

# Quick local test o verify the loader tests
if __name__ == "__main__":
    dataset = FastConformerDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch_features, batch_texts in dataloader:
        print("\n--- TEST BATCH LOADED ---")
        print(f"Batch Tensor Shape: {batch_features.shape}")
        print(f"Batch Transcripts:  {batch_texts}")
        print("Success: The PyTorch DataLoader is ready for training!")
        break
