"""Silero VAD loader (pulled from Torch Hub)."""
import torch


def load_vad_model():
    """Load Silero VAD from Torch Hub. Blocks on first call (network + disk)."""
    print("Loading Silero VAD model...")
    model, _utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    print("VAD model ready.")
    return model
