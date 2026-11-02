import warnings

# pyannote warns at import time when torchcodec cannot load FFmpeg (e.g. FFmpeg 9).
# We always pass in-memory waveforms, so that decoder is never used.
warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly")

from .schema import SpeakerAssignedWord, SpeakerSegment, TranscribedWord  # noqa: E402


def __getattr__(name):
    # Preserve the public import API without loading optional audio packages for
    # remote-only desktop/MCP clients.
    if name in {"ProgressCallback", "TranscriptionWorker"}:
        from . import worker

        return getattr(worker, name)
    raise AttributeError(name)


__all__ = [
    "ProgressCallback",
    "SpeakerAssignedWord",
    "SpeakerSegment",
    "TranscribedWord",
    "TranscriptionWorker",
]
