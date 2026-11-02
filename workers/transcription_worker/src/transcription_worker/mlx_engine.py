"""Native Apple Silicon engine; timestamps stay on the original recording."""

from pathlib import Path
from collections.abc import Callable

from .models import WHISPER_MODEL_IDS, WhisperModelType
from .schema import TranscribedWord


class MLXWhisper:
    def __init__(self, model_type: WhisperModelType, on_state: Callable[[str], None]):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder
        from huggingface_hub import snapshot_download

        self.model_type = model_type
        self.device = "mlx"
        self.repo = (
            "mlx-community/whisper-turbo"
            if model_type == WhisperModelType.TURBO
            else f"mlx-community/whisper-{WHISPER_MODEL_IDS[model_type]}-mlx"
        )
        on_state("downloading_model")
        self.path = snapshot_download(self.repo)
        on_state("loading_models")
        ModelHolder.get_model(self.path, mx.float16)

    def transcribe(
        self, audio_file_path: str | Path, *, language=None, on_progress=None
    ):
        import mlx_whisper

        # No silence concatenation: MLX consumes the original WAV and its native
        # word times already include every silence gap.
        output = mlx_whisper.transcribe(
            str(audio_file_path),
            path_or_hf_repo=self.path,
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
            verbose=None,
        )
        words = []
        for segment in output["segments"]:
            for word in segment.get("words", []):
                if word["word"].strip():
                    words.append(
                        TranscribedWord(
                            word=word["word"].strip(),
                            start_time_seconds=word["start"],
                            end_time_seconds=word["end"],
                            probablility=word.get("probability"),
                        )
                    )
            if on_progress:
                on_progress(segment["end"])
        return sorted(words, key=lambda word: word.start_time_seconds)
