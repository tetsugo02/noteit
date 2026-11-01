import warnings

# pyannote warns at import time when torchcodec cannot load FFmpeg (e.g. FFmpeg 9).
# We always pass in-memory waveforms, so that decoder is never used.
warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly")

from .schema import SpeakerAssignedWord, SpeakerSegment, TranscribedWord  # noqa: E402
from .worker import ProgressCallback, TranscriptionWorker  # noqa: E402

__all__ = [
    "ProgressCallback",
    "SpeakerAssignedWord",
    "SpeakerSegment",
    "TranscribedWord",
    "TranscriptionWorker",
]
