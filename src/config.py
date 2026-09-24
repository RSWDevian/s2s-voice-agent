# Path, hyperparams and device settings
import os
import torch
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_AUDIO_DIR = os.path.join(DATA_DIR, 'raw_audio')
PROCESSED_TENSORS_DIR = os.path.join(DATA_DIR, 'processed_tensors')
TRAINED_MODELS_DIR = os.path.join(DATA_DIR, 'trained_models')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'models', 'local_checkpoints')
ADAPTERS_DIR = os.path.join(BASE_DIR, 'models', 'adapters')

# Hardware Detection
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# LLM backbone
LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_DIM = 896  # hidden_size of LLM_ID

# Training data
DATASET_ID = "agarwalayushi/hinglish"  # HF dataset: audio + text pairs
AUDIO_SAMPLE_RATE = 16000  # FastConformer's expected input rate
MIMI_SAMPLE_RATE = 24000  # Mimi codec's native rate (models/local_checkpoints/mimi_codec/config.json)