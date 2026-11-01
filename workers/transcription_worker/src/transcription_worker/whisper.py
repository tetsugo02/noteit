from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines import automatic_speech_recognition as _asr_pipeline
from transformers.pipelines.base import Pipeline

from .models import WHISPER_MODEL_IDS, WhisperModelType, resolve_device
from .schema import TranscribedWord
from .utils.audio_io import read_mono

__all__ = ["Whisper", "WhisperModelType"]


def _torchcodec_loads() -> bool:
    """True only if torchcodec's native libraries actually load.

    transformers imports torchcodec whenever the package is installed, even for
    in-memory input, and crashes when it cannot find a supported FFmpeg (e.g.
    Homebrew FFmpeg 9). We always pass arrays, so treat a broken install as absent.
    """
    try:
        import torchcodec.decoders  # noqa: F401
    except Exception:  # noqa: BLE001 - OSError/RuntimeError from dlopen
        return False
    return True


_TORCHCODEC_LOADS = _torchcodec_loads()
_asr_pipeline.is_torchcodec_available = lambda: _TORCHCODEC_LOADS  # type: ignore[attr-defined]


class Whisper:
    """Long-lived Whisper inference service used by a transcription worker."""

    def __init__(
        self,
        hf_token: str | None = None,
        model_type: WhisperModelType = WhisperModelType.TURBO,
        device: str | None = None,
        compile_model: bool = True,
    ) -> None:
        self.device = resolve_device(device or "auto")
        self.model_type = model_type

        # float16 on MPS can produce NaNs/garbage with some Whisper checkpoints,
        # so only CUDA uses half precision.
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        model_id = WHISPER_MODEL_IDS[model_type]

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=dtype,
            token=hf_token,
            use_safetensors=True,
        )
        self.model.to(self.device)

        self.is_compiled = compile_model and self.device.startswith("cuda")
        if self.is_compiled:
            torch.set_float32_matmul_precision("high")
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(
                self.model.forward,
                # Whisper's static KV cache mutates its inputs, so CUDA Graphs
                # used by reduce-overhead are not applicable here.
                mode="default",
                # Transformers 5.x output capture uses locks and ContextVar.
                # Word timestamps therefore require graph breaks around that code.
                fullgraph=False,
            )

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
            "return_timestamps": "word",
            "decoder_kwargs": {"clean_up_tokenization_spaces": False},
        }
        if language is not None:
            inference_options["language"] = language

        samples, sample_rate = read_mono(file_path)
        audio = {"raw": samples, "sampling_rate": sample_rate}
        if self.is_compiled:
            with sdpa_kernel(SDPBackend.MATH):
                raw_result = self.pipe(audio, **inference_options)
        else:
            raw_result = self.pipe(audio, **inference_options)
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
