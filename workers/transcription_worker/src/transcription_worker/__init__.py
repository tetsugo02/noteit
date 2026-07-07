from .schema import SpeakerAssignedWord, SpeakerSegment, TranscribedWord
from .worker import ProgressCallback, TranscriptionWorker


def main() -> None:
    print("Hello from stranscription-worker!")

__all__ = [
    "ProgressCallback",
    "SpeakerAssignedWord",
    "SpeakerSegment",
    "TranscribedWord",
    "TranscriptionWorker",
]
