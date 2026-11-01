from pathlib import Path
from typing import Any, cast

import torch

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput

from transcription_worker.utils.audio_format_concert import convert_audio_format

from .models import DIARIZER_MODEL_IDS, DiarizerModelType, resolve_device
from .schema import SpeakerSegment
from .utils.audio_io import read_mono

__all__ = ["DiarizerModelType", "SpeakerDiarizer"]


class SpeakerDiarizer:
    def __init__(
        self,
        hf_token: str,
        model_type: DiarizerModelType,
        device: str | None = None,
    ) -> None:
        # pyannote on MPS is unreliable, so "auto" only picks CUDA or CPU.
        self.device = resolve_device(device or "auto", allow_mps=False)
        self.model_type = model_type
        pipeline = Pipeline.from_pretrained(
            DIARIZER_MODEL_IDS[model_type],
            token=hf_token,
        )
        if pipeline is None:
            raise RuntimeError(
                f"Could not load {DIARIZER_MODEL_IDS[model_type]}. Accept the model "
                "terms on Hugging Face and check the access token."
            )
        pipeline.to(torch.device(self.device))
        self.diarizer_pipline = pipeline

    def diarize(
        self,
        file_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[SpeakerSegment]:
        """
        Diarize the given audio file and return the diarization output.
        """
        if not self.diarizer_pipline:
            raise ValueError("Diarization pipeline is not initialized.")

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # check file format
        if file_path.suffix.lower() != ".wav":
            # convert to wav
            wav_file_path = file_path.with_suffix(".wav")
            convert_audio_format(file_path, wav_file_path)
            file_path = wav_file_path

        speaker_hints: dict[str, Any] = {
            key: value
            for key, value in {
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }.items()
            if value is not None
        }
        samples, sample_rate = read_mono(file_path)
        audio = {
            "waveform": torch.from_numpy(samples).unsqueeze(0),
            "sample_rate": sample_rate,
        }
        output = self.diarizer_pipline(audio, **speaker_hints)
        output = cast(DiarizeOutput, output)

        diarization = output.speaker_diarization

        result: list[SpeakerSegment] = []

        for turn, speaker in diarization:  # type: ignore
            speaker = cast(str, speaker)
            segment = SpeakerSegment(
                start_time=turn.start, end_time=turn.end, speaker_label=speaker
            )
            result.append(segment)

        return result
