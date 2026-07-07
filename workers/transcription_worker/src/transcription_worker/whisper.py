from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.base import Pipeline

from .schema import TranscribedWord


class WhisperModelType(StrEnum):
    """Stable model names that can be exposed by the worker API."""

    TURBO = "turbo"
    LARGE = "large"
    MEDIUM = "medium"


WHISPER_MODEL_IDS: dict[WhisperModelType, str] = {
    WhisperModelType.TURBO: "openai/whisper-large-v3-turbo",
    WhisperModelType.LARGE: "openai/whisper-large-v3",
    WhisperModelType.MEDIUM: "openai/whisper-medium",
}


class Whisper:
    """Long-lived Whisper inference service used by a transcription worker."""

    def __init__(
        self,
        hf_token: str | None = None,
        model_type: WhisperModelType = WhisperModelType.TURBO,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available.")

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        model_id = WHISPER_MODEL_IDS[model_type]

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=dtype,
            token=hf_token,
            use_safetensors=True,
        )
        self.model.to(self.device)

        processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        self.pipe: Pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=self.device,
            dtype=dtype,
        )

    def transcribe(
        self,
        audio_file_path: str | Path,
        *,
        language: str | None = None,
    ) -> list[TranscribedWord]:
        """Transcribe an audio file into the worker's word-level schema."""
        file_path = Path(audio_file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        inference_options: dict[str, Any] = {
            "chunk_length_s": 30,
            "return_timestamps": "word",
        }
        if language is not None:
            inference_options["language"] = language

        raw_result = self.pipe(str(file_path), **inference_options)
        return self._to_transcribed_words(raw_result)

    @staticmethod
    def _to_transcribed_words(raw_result: object) -> list[TranscribedWord]:
        """Validate and isolate the weakly typed Transformers result boundary."""
        if not isinstance(raw_result, Mapping):
            raise TypeError("Whisper returned an unexpected result type.")

        chunks = raw_result.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError("Whisper result does not contain word timestamps.")

        words: list[TranscribedWord] = []
        for raw_chunk in chunks:
            if not isinstance(raw_chunk, Mapping):
                raise TypeError("Whisper returned an invalid word chunk.")

            text = raw_chunk.get("text")
            timestamp = raw_chunk.get("timestamp")
            if (
                not isinstance(text, str)
                or not isinstance(timestamp, (tuple, list))
                or len(timestamp) != 2
                or not all(isinstance(value, (int, float)) for value in timestamp)
            ):
                raise ValueError("Whisper returned an invalid word timestamp.")

            start_time, end_time = cast(tuple[float, float], tuple(timestamp))
            words.append(
                TranscribedWord(
                    word=text.strip(),
                    start_time_seconds=float(start_time),
                    end_time_seconds=float(end_time),
                    # Transformers does not expose a calibrated word probability.
                    probablility=None,
                )
            )

        return words
