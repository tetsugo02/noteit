from enum import StrEnum
from pathlib import Path
from typing import cast

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput

from transcription_worker.utils.audio_format_concert import convert_audio_format

from .schema import SpeakerSegment


class DiarizerModelType(StrEnum):
    COMMUNITY = "community"
    STANDARD = "standard"


DIARIZER_MODEL_IDS: dict[DiarizerModelType, str] = {
    DiarizerModelType.COMMUNITY: "pyannote/speaker-diarization-community-1",
    DiarizerModelType.STANDARD: "pyannote/speaker-diarization-3.1",
}


class SpeakerDiarizer:
    def __init__(
        self, hf_token: str, model_type: DiarizerModelType, device: str = "cuda"
    ) -> None:
        self.diarizer_pipline = Pipeline.from_pretrained(
            DIARIZER_MODEL_IDS[model_type],
            token=hf_token,
        )

    def diarize(self, file_path: Path) -> list[SpeakerSegment]:
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

        output = self.diarizer_pipline(file_path)
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
