"""Model choices and compute-device selection shared by the CLI and MCP server."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


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
    WhisperModelType.TINY: "tiny",
    WhisperModelType.BASE: "base",
    WhisperModelType.SMALL: "small",
    WhisperModelType.MEDIUM: "medium",
    WhisperModelType.TURBO: "turbo",
    WhisperModelType.LARGE: "large-v3",
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

DEVICE_CHOICES = ("auto", "mlx", "cuda", "cpu")


def resolve_device(requested: str = "auto", *, allow_mps: bool = True) -> str:
    """Pick a torch device. ``auto`` prefers CUDA, then Apple MPS, then CPU."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        import ctranslate2

        cuda = ctranslate2.get_cuda_device_count() > 0
        if requested == "auto":
            return "cuda" if cuda else "cpu"
        if requested.startswith("cuda") and cuda:
            return requested
        raise RuntimeError(f"Device unavailable: {requested}")

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
    try:
        import ctranslate2

        cuda = ctranslate2.get_cuda_device_count() > 0
    except (ImportError, RuntimeError):
        cuda = False
    return WhisperModelType.TURBO if cuda else WhisperModelType.SMALL


def recognition_device(requested: str = "auto") -> tuple[str, str]:
    """Resolve an engine without importing inference libraries on the frontend."""
    import platform
    import importlib.util

    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    mlx_available = importlib.util.find_spec("mlx_whisper") is not None
    if requested == "mlx" or (requested == "auto" and apple and mlx_available):
        if not apple or not mlx_available:
            raise RuntimeError(
                "Apple GPU requires Apple Silicon and the mac-gpu extra."
            )
        return "mlx-whisper", "mlx"
    return "faster-whisper", resolve_device(requested, allow_mps=False)


def engine_capabilities() -> dict:
    import importlib.util
    import platform

    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    # CTranslate2 can probe CUDA without loading model weights or torch.
    devices = []
    if importlib.util.find_spec("faster_whisper"):
        devices.append("cpu")
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count():
                devices.append("cuda")
        except (ImportError, RuntimeError):
            pass
    engines = [{"name": "faster-whisper", "devices": devices}] if devices else []
    if apple and importlib.util.find_spec("mlx_whisper"):
        engines.append({"name": "mlx-whisper", "devices": ["mlx"]})
    return {
        "engines": engines,
        "devices": [device for engine in engines for device in engine["devices"]],
        "diarization": importlib.util.find_spec("pyannote") is not None,
    }
