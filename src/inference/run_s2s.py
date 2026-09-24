import argparse
import os
import sys

import soundfile as sf
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import MIMI_SAMPLE_RATE
from src.models.s2s_composite import S2SModel


def main():
    parser = argparse.ArgumentParser(description="Speech-to-speech smoke-test CLI.")
    parser.add_argument("input_audio", help="Path to input .wav (any sample rate, mono or stereo).")
    parser.add_argument("output_audio", help="Path to write the generated .wav response (24kHz).")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    audio, sample_rate = sf.read(args.input_audio, dtype="float32", always_2d=False)
    if audio.ndim > 1:  # collapse stereo -> mono
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio)

    kwargs = {"temperature": args.temperature, "top_k": args.top_k}
    if args.max_new_tokens is not None:
        kwargs["max_new_tokens"] = args.max_new_tokens
    model = S2SModel(**kwargs)

    print(f"[*] Running S2S inference on {args.input_audio} ...")
    output = model.speak(waveform, sample_rate)
    sf.write(args.output_audio, output.numpy(), MIMI_SAMPLE_RATE)
    print(f"[+] Wrote {output.shape[0] / MIMI_SAMPLE_RATE:.2f}s of audio to {args.output_audio}")


if __name__ == "__main__":
    main()
