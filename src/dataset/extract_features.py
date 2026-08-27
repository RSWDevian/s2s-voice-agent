# Offline script: .wav -> FastConformer -> .pt
import os
import sys
import torch
from tqdm import tqdm
from datasets import load_dataset
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.config import CHECKPOINTS_DIR, PROCESSED_TENSORS_DIR, DEVICE
os.makedirs(PROCESSED_TENSORS_DIR, exist_ok=True)

class FastConformerExtractor:
    def __init__(self):
        print(f"[*] Loading cached FastConformer model ...")

        nemo_cache = os.path.join(CHECKPOINTS_DIR, "nemo_cache")
        os.environ['NEMO_CACHE_DIR'] = nemo_cache

        import nemo.collections.asr as nemo_asr
        model_name = "nvidia/stt_en_fastconformer_hybrid_large_pc"
        local_checkpoint = os.path.join(nemo_cache, "stt_en_fastconformer_hybrid_large_pc.nemo")
        model_class = nemo_asr.models.EncDecHybridRNNTCTCBPEModel
        if os.path.isfile(local_checkpoint):
            self.model = model_class.restore_from(
                restore_path=local_checkpoint,
                map_location=DEVICE,
            )
        else:
            self.model = model_class.from_pretrained(model_name=model_name)
        self.model = self.model.to(DEVICE)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"[*] FastConformer model loaded successfully.")

    @torch.no_grad()
    def extract_from_waveform(self, audio_array: torch.Tensor, sample_rate: int = 16000):
        """
        Passes the 16kHz waveform through the 8x sampling convolutional encoder
        """
        if audio_array.ndim == 1:
            audio_array = audio_array.unsqueeze(0)

        audio_array = audio_array.to(DEVICE)
        audio_length = torch.tensor([audio_array.shape[1]]).to(DEVICE)

        # Converting the audio wave to mel-spectrogram
        processed_signal, processed_length = self.model.preprocessor(
            input_signal=audio_array,
            length=audio_length
        )

        # Extract acoustic embeddings via FastConformer encoder
        encoded_audio, encoded_len = self.model.encoder(
            audio_signal=processed_signal,
            length=processed_length
        )

        return encoded_audio.transpose(1,2).cpu()

def run_feature_extraction(dataset_name: str = ""):
    manifest_path = os.path.join(PROCESSED_TENSORS_DIR, "manifest.pt")

    # Resume logic
    if os.path.exists(manifest_path):
        manifest = torch.load(manifest_path)
        start_index = len(manifest)
        print(f"[*] Resuming feature extraction from index {start_index} ...")
    else:
        manifest = []
        start_index = 0
        print(f"[*] Starting feature extraction from scratch ...")
    
    print(f"[*] Extracting FastConformer features samples from Hugging Face ...")
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    if start_index > 0:
        dataset = dataset.skip(start_index)
    extractor = FastConformerExtractor()

    try:
        for i, items in enumerate(tqdm(dataset, desc="Extracting audio", initial=start_index), start=start_index):
            raw_audio = items["audio"]["array"]
            audio_tensor = torch.tensor(raw_audio, dtype=torch.float32)
            id = f"processed_audio{i:06d}"
            tensor_filename = f"{id}.pt"
            tensor_filepath = os.path.join(PROCESSED_TENSORS_DIR, tensor_filename)

            features = extractor.extract_from_waveform(audio_tensor)
            torch.save(features.squeeze(0), tensor_filepath)

            manifest.append({
                "id": id,
                "features_path": tensor_filepath,
                "text_transcript": items["text"].strip().lower(),
            })

            if (i + 1) % 100 == 0:
                torch.save(manifest, manifest_path)
    except KeyboardInterrupt:
        print(f"\n[!] Process paused by user (Ctrl + C)...")
    finally:
        torch.save(manifest, manifest_path)
        print(f"[*] Extraction saved successfully")
        print(f"[*] Total precomputed tensors in {PROCESSED_TENSORS_DIR}: {len(manifest)}")



if __name__ == "__main__":
    run_feature_extraction(dataset_name="agarwalayushi/hinglish")