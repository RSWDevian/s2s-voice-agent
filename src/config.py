# Path, hyperparams and device settings
import os
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_AUDIO_DIR = os.path.join(DATA_DIR, 'raw_audio')
PROCESSED_TENSORS_DIR = os.path.join(DATA_DIR, 'processed_tensors')
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