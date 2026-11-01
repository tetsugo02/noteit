"""Decode WAV files with libsndfile instead of torchcodec.

transformers and pyannote decode file paths through torchcodec, which only
loads against specific FFmpeg major versions (e.g. it fails with Homebrew's
FFmpeg 9). Reading WAV ourselves and passing in-memory audio avoids that.
"""

from pathlib import Path

import numpy as np
import soundfile


def read_mono(file_path: Path) -> tuple[np.ndarray, int]:
    """Return float32 mono samples and the sample rate."""
    samples, sample_rate = soundfile.read(file_path, dtype="float32", always_2d=True)
    return np.ascontiguousarray(samples.mean(axis=1)), int(sample_rate)
