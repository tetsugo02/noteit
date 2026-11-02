from collections.abc import Callable
from pathlib import Path
import warnings

from faster_whisper import WhisperModel

from .models import WHISPER_MODEL_IDS, WhisperModelType, resolve_device
from .schema import TranscribedWord

__all__ = ["Whisper", "WhisperModelType"]


class Whisper:
    """Long-lived faster-whisper model with VAD and global word timestamps."""

    def __init__(
        self,
        hf_token: str | None = None,
        model_type: WhisperModelType = WhisperModelType.TURBO,
        device: str | None = None,
        compile_model: bool = False,
        on_state: Callable[[str], None] | None = None,
    ) -> None:
        if device == "mps":
            raise ValueError("faster-whisper does not support MPS. Use cpu or auto.")
        self.device = resolve_device(device or "auto", allow_mps=False)
        self.model_type = model_type
        if compile_model:
            warnings.warn(
                "compile_model is ignored by faster-whisper (CTranslate2).",
                DeprecationWarning,
                stacklevel=2,
            )
        device_name, _, device_index = self.device.partition(":")
        model_path = WHISPER_MODEL_IDS[model_type]
        if on_state:
            from faster_whisper.utils import download_model

            on_state("downloading_model")
            model_path = download_model(model_path, use_auth_token=hf_token)
            on_state("loading_models")
        self.model = WhisperModel(
            model_path,
            device=device_name,
            device_index=int(device_index or 0),
            compute_type="float16" if device_name == "cuda" else "int8",
            use_auth_token=hf_token,
        )

    def transcribe(
        self,
        audio_file_path: str | Path,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[TranscribedWord]:
        """Recognize the entire recording once, skipping silence with VAD.

        Progress reports seconds in the original recording, including VAD gaps.
        No phrase blacklist is used: stock phrases can be real meeting speech.
        """
        file_path = Path(audio_file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        segments, info = self.model.transcribe(
            str(file_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
        )
        words: list[TranscribedWord] = []
        for segment in segments:
            for word in segment.words or []:
                text = word.word.strip()
                if text:
                    words.append(
                        TranscribedWord(
                            word=text,
                            start_time_seconds=word.start,
                            end_time_seconds=word.end,
                            probablility=word.probability,
                        )
                    )
            if on_progress is not None:
                on_progress(segment.end)
        if on_progress is not None:
            on_progress(info.duration)
        return sorted(words, key=lambda word: word.start_time_seconds)
