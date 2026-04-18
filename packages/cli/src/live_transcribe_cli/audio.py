"""Audio capture and device discovery. CLI-only."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from live_transcribe_core.config import SAMPLE_RATE


def find_blackhole_device() -> tuple[Optional[int], Optional[str]]:
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "blackhole" in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i, dev["name"]
    return None, None


def list_input_devices(default_idx: Optional[int] = None) -> None:
    devices = sd.query_devices()
    print("\nAvailable input devices:")
    print("-" * 60)
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            markers = []
            if "blackhole" in dev["name"].lower():
                markers.append("BlackHole")
            if i == default_idx:
                markers.append("default")
            suffix = f"  <-- [{', '.join(markers)}]" if markers else ""
            print(f"  [{i}] {dev['name']} ({dev['max_input_channels']}ch, "
                  f"{dev['default_samplerate']:.0f}Hz){suffix}")
    print()


def open_stream(device_idx: int, on_chunk: Callable[[np.ndarray], None]) -> sd.InputStream:
    """Create a float32 mono @ 16 kHz InputStream that forwards chunks to the engine."""
    return sd.InputStream(
        device=device_idx,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * 0.1),
        callback=lambda indata, frames, time_info, status: on_chunk(indata[:, 0].copy()),
    )
