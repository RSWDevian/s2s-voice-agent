import os
import logging
import sys
from abc import ABC, abstractmethod
from huggingface_hub import snapshot_download

# Support direct script execution (`python src/utils/download_weights.py`) by
# ensuring the repository root is on sys.path before importing `src.*`.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import CHECKPOINTS_DIR

# clean logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BaseModelDownloader(ABC):
    """
    Abstract base class for model downloaders.
    """
    def __init__(self, model_name: str, folder_name: str, cache_dir: str = CHECKPOINTS_DIR):
        self.model_name = model_name
        self.folder_name = folder_name
        self.cache_dir = cache_dir
        self.target_path = os.path.join(self.cache_dir, self.folder_name)
        # Ensure base cache dir exists
        os.makedirs(self.target_path, exist_ok=True)

    def _download_huggingface_model(self):
        """
        Reusable method to download a model from Hugging Face Hub.
        """
        logger.info(f"Checking/Downloading model '{self.model_name}' to '{self.target_path}'...")
        try:
            snapshot_download(
                repo_id=self.model_name,
                cache_dir=self.cache_dir,
                local_dir=self.target_path,
                local_dir_use_symlinks=False
            )
            logger.info(f"Model '{self.model_name}' downloaded successfully to '{self.target_path}'.")
        except Exception as e:
            logger.error(f"Failed to download model '{self.model_name}': {e}")
            raise
        return self.target_path

    @abstractmethod
    def download_model(self):
        pass

class HuggingFaceModelDownloader(BaseModelDownloader):
    """
    Concrete implementation for downloading models from Hugging Face Hub.
    """
    def download_model(self):
        return self._download_huggingface_model()


class NeMoModelDownloader(BaseModelDownloader):
    """
    Concrete implementation for downloading NeMo models.
    """
    def download_model(self):
        logger.info(f"Checking/Downloading NeMo model '{self.model_name}' to '{self.target_path}'...")

        # Redirect NeMo's internal cache directory into our project repository
        nemo_cache = self.target_path
        os.environ['NEMO_CACHE_DIR'] = nemo_cache

        # NeMo imports NeptuneLogger from pytorch_lightning.loggers, but newer
        # pytorch-lightning builds may not expose that symbol. Provide a small
        # fallback so NeMo can import and proceed when Neptune logging is unused.
        import pytorch_lightning.loggers as pl_loggers
        if not hasattr(pl_loggers, 'NeptuneLogger'):
            class NeptuneLogger:  # pragma: no cover - import compatibility shim
                def __init__(self, *args, **kwargs):
                    raise RuntimeError(
                        "NeptuneLogger is unavailable in this pytorch-lightning version. "
                        "Install a compatible version if Neptune logging is required."
                    )

            pl_loggers.NeptuneLogger = NeptuneLogger
        
        # Deferred import to ensure env vars are applied
        import nemo.collections.asr as nemo_asr
        
        # NeMo may resolve through the global HF cache; explicitly save a local
        # .nemo artifact in our project cache directory for predictable paths.
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=self.model_name)
        local_nemo_file = os.path.join(
            nemo_cache,
            f"{self.model_name.split('/')[-1]}.nemo"
        )
        model.save_to(local_nemo_file)
        logger.info(f"NeMo model artifact saved to: {local_nemo_file}")
        
        logger.info(f"NeMo model ready at cache: {nemo_cache}")
        return nemo_cache

def download_all_models():
    download_registry = [
        # The reasoning backbone
        HuggingFaceModelDownloader(
            model_name="Qwen/Qwen2.5-1.5B",
            folder_name="qwen_backbone"
        ),
        # Audio codec /vocoder model
        HuggingFaceModelDownloader(
            model_name="Kyutai/mimi",
            folder_name="mimi_codec"
        ),
        # The Audio Encoder
        NeMoModelDownloader(
            model_name="nvidia/stt_en_fastconformer_hybrid_large_pc",
            folder_name="nemo_cache"
        )
    ]

    logger.info("=== Starting model download & local caching process ===")
    for downloader in download_registry:
        downloader.download_model()
    logger.info("=== All models downloaded and cached successfully ===")

if __name__ == "__main__":
    download_all_models()