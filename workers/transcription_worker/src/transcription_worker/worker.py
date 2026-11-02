from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TYPE_CHECKING

from pydub import AudioSegment

from .schema import SpeakerAssignedWord, SpeakerSegment, TranscribedWord

if TYPE_CHECKING:
    from .speaker_diarizer import SpeakerDiarizer
    from .whisper import Whisper

ProgressCallback = Callable[[SpeakerSegment], None]
DiarizedCallback = Callable[[list[SpeakerSegment]], None]


def assign_speakers(
    words: list[TranscribedWord], segments: list[SpeakerSegment]
) -> list[SpeakerAssignedWord]:
    """Choose the speaker with the greatest total time overlap per word.

    Ties use the first chronological turn. Uncovered words keep their text
    with SPEAKER_UNKNOWN rather than guessing a speaker during silence.
    """
    turns = sorted(segments, key=lambda segment: segment.start_time)
    result: list[SpeakerAssignedWord] = []
    active: list[SpeakerSegment] = []
    index = 0
    for word in sorted(words, key=lambda word: word.start_time_seconds):
        start, end = word.start_time_seconds, word.end_time_seconds
        while index < len(turns) and turns[index].start_time <= end:
            active.append(turns[index])
            index += 1
        active = [turn for turn in active if turn.end_time > start]
        overlap: dict[str, float] = {}
        for turn in active:
            duration = max(0.0, min(end, turn.end_time) - max(start, turn.start_time))
            if end == start and turn.start_time <= start < turn.end_time:
                duration = 1.0
            if duration > 0:
                overlap[turn.speaker_label] = (
                    overlap.get(turn.speaker_label, 0) + duration
                )
        result.append(
            SpeakerAssignedWord(
                **word.model_dump(),
                speaker_label=max(overlap, key=overlap.__getitem__)
                if overlap
                else "SPEAKER_UNKNOWN",
            )
        )
    return result


class TranscriptionWorker:
    """Backend-facing, long-lived diarization and transcription pipeline.

    Construct this class once when the worker process starts. Calling
    ``transcribe`` is synchronous and GPU access is serialized, so an async
    backend should invoke it through its job queue rather than on its event loop.
    """

    def __init__(self, diarizer: SpeakerDiarizer | None, whisper: Whisper) -> None:
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
        with TemporaryDirectory(
            prefix="noteit-transcription-", dir=os.environ.get("NOTEIT_TEMP_DIR")
        ) as temporary_directory:
            temporary_path = Path(temporary_directory)

            # Decode once to 16 kHz mono WAV, which both pyannote and Whisper
            # expect, instead of writing a converted file next to the input.
            audio = (
                # Let ffmpeg downmix/resample while decoding: a 70 min 48 kHz
                # stereo file would otherwise take ~800 MB as raw PCM.
                AudioSegment.from_file(
                    audio_file_path, parameters=["-ac", "1", "-ar", "16000"]
                )
                .set_channels(1)
                .set_frame_rate(16_000)
            )
            wav_path = temporary_path / "input.wav"
            audio.export(wav_path, format="wav")

            speaker_segments = (
                self._diarizer.diarize(wav_path, **speaker_hints)
                if self._diarizer
                else []
            )
            if on_diarized is not None:
                on_diarized(speaker_segments)

            # Preserve the progress API while recognizing the recording once.
            pending = iter(sorted(speaker_segments, key=lambda turn: turn.end_time))
            next_turn = next(pending, None)

            def on_progress(seconds: float) -> None:
                nonlocal next_turn
                while next_turn is not None and next_turn.end_time <= seconds:
                    if on_segment is not None:
                        on_segment(next_turn)
                    next_turn = next(pending, None)

            words = self._whisper.transcribe(
                wav_path, language=language, on_progress=on_progress
            )
            on_progress(float("inf"))
            return (
                assign_speakers(words, speaker_segments)
                if self._diarizer
                else [
                    SpeakerAssignedWord(**word.model_dump(), speaker_label=None)
                    for word in words
                ]
            )
