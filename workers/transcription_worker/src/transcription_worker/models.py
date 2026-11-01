"""Model choices and compute-device selection shared by the CLI and MCP server."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class WhisperModelType(StrEnum):
    """Stable Whisper model names, ordered from lightest to heaviest."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    TURBO = "turbo"
    LARGE = "large"


class DiarizerModelType(StrEnum):
    COMMUNITY = "community"
    STANDARD = "standard"


WHISPER_MODEL_IDS: dict[WhisperModelType, str] = {
    WhisperModelType.TINY: "openai/whisper-tiny",
    WhisperModelType.BASE: "openai/whisper-base",
    WhisperModelType.SMALL: "openai/whisper-small",
    WhisperModelType.MEDIUM: "openai/whisper-medium",
    WhisperModelType.TURBO: "openai/whisper-large-v3-turbo",
    WhisperModelType.LARGE: "openai/whisper-large-v3",
}

DIARIZER_MODEL_IDS: dict[DiarizerModelType, str] = {
    DiarizerModelType.COMMUNITY: "pyannote/speaker-diarization-community-1",
    DiarizerModelType.STANDARD: "pyannote/speaker-diarization-3.1",
}


@dataclass(frozen=True)
class ModelInfo:
    parameters: str
    note: str


WHISPER_MODEL_INFO: dict[WhisperModelType, ModelInfo] = {
    WhisperModelType.TINY: ModelInfo(
        "39M", "Fastest. Rough Japanese accuracy; for testing."
    ),
    WhisperModelType.BASE: ModelInfo("74M", "Very fast. Usable for clear speech only."),
    WhisperModelType.SMALL: ModelInfo("244M", "Good default for laptops without CUDA."),
    WhisperModelType.MEDIUM: ModelInfo("769M", "Noticeably better accuracy, slower."),
    WhisperModelType.TURBO: ModelInfo(
        "809M", "Near-large accuracy at much higher speed."
    ),
    WhisperModelType.LARGE: ModelInfo(
        "1.55B", "Best accuracy. Needs a GPU in practice."
    ),
}

DIARIZER_MODEL_INFO: dict[DiarizerModelType, ModelInfo] = {
    DiarizerModelType.COMMUNITY: ModelInfo("-", "pyannote community-1. Recommended."),
    DiarizerModelType.STANDARD: ModelInfo("-", "pyannote 3.1. Older pipeline."),
}

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def resolve_device(requested: str = "auto", *, allow_mps: bool = True) -> str:
    """Pick a torch device. ``auto`` prefers CUDA, then Apple MPS, then CPU."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if allow_mps and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but MPS is not available.")
    return requested


def default_whisper_model() -> WhisperModelType:
    """Turbo on CUDA machines, small elsewhere (e.g. a MacBook)."""
    return (
        WhisperModelType.TURBO if torch.cuda.is_available() else WhisperModelType.SMALL
    )
