from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TYPE_CHECKING

from pydub import AudioSegment

from .schema import SpeakerAssignedWord, SpeakerSegment

if TYPE_CHECKING:
    from .speaker_diarizer import SpeakerDiarizer
    from .whisper import Whisper

ProgressCallback = Callable[[SpeakerSegment], None]
DiarizedCallback = Callable[[list[SpeakerSegment]], None]


class TranscriptionWorker:
    """Backend-facing, long-lived diarization and transcription pipeline.

    Construct this class once when the worker process starts. Calling
    ``transcribe`` is synchronous and GPU access is serialized, so an async
    backend should invoke it through its job queue rather than on its event loop.
    """

    def __init__(self, diarizer: SpeakerDiarizer, whisper: Whisper) -> None:
        self._diarizer = diarizer
        self._whisper = whisper
        self._inference_lock = Lock()

    def transcribe(
        self,
        audio_file_path: str | Path,
        *,
        language: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        on_segment: ProgressCallback | None = None,
        on_diarized: DiarizedCallback | None = None,
    ) -> list[SpeakerAssignedWord]:
        """Return schema-compatible, speaker-assigned words for one audio file.

        Any format ffmpeg can read (m4a, mp3, mp4, ...) is accepted. Speaker
        count hints are forwarded to pyannote and improve diarization when known.
        """
        file_path = Path(audio_file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        speaker_hints = {
            key: value
            for key, value in {
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }.items()
            if value is not None
        }

        with self._inference_lock:
            return self._transcribe(
                file_path,
                language=language,
                speaker_hints=speaker_hints,
                on_segment=on_segment,
                on_diarized=on_diarized,
            )

    def _transcribe(
        self,
        audio_file_path: Path,
        *,
        language: str | None,
        speaker_hints: dict[str, int],
        on_segment: ProgressCallback | None,
        on_diarized: DiarizedCallback | None,
    ) -> list[SpeakerAssignedWord]:
        result: list[SpeakerAssignedWord] = []

        with TemporaryDirectory(prefix="noteit-transcription-") as temporary_directory:
            temporary_path = Path(temporary_directory)

            # Decode once to 16 kHz mono WAV, which both pyannote and Whisper
            # expect, instead of writing a converted file next to the input.
            audio = (
                AudioSegment.from_file(audio_file_path)
                .set_channels(1)
                .set_frame_rate(16_000)
            )
            wav_path = temporary_path / "input.wav"
            audio.export(wav_path, format="wav")

            speaker_segments = self._diarizer.diarize(wav_path, **speaker_hints)
            if on_diarized is not None:
                on_diarized(speaker_segments)

            for index, segment in enumerate(speaker_segments):
                start_ms = max(0, round(segment.start_time * 1000))
                end_ms = min(len(audio), round(segment.end_time * 1000))
                if end_ms <= start_ms:
                    continue

                if on_segment is not None:
                    on_segment(segment)

                segment_path = temporary_path / f"segment-{index:04d}.wav"
                audio[start_ms:end_ms].export(segment_path, format="wav")
                words = self._whisper.transcribe(segment_path, language=language)

                result.extend(
                    SpeakerAssignedWord(
                        word=word.word,
                        start_time_seconds=segment.start_time + word.start_time_seconds,
                        end_time_seconds=segment.start_time + word.end_time_seconds,
                        probablility=word.probablility,
                        speaker_label=segment.speaker_label,
                    )
                    for word in words
                )

        return result
