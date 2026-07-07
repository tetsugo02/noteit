from pathlib import Path
from unittest.mock import Mock

from pydub import AudioSegment
from transcription_worker.schema import SpeakerSegment, TranscribedWord
from transcription_worker.worker import TranscriptionWorker


def test_transcribe_assigns_speaker_and_converts_to_global_timestamps(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    AudioSegment.silent(duration=3000).export(audio_path, format="wav")

    diarizer = Mock()
    diarizer.diarize.return_value = [
        SpeakerSegment(start_time=1.0, end_time=2.5, speaker_label="SPEAKER_00")
    ]
    whisper = Mock()
    whisper.transcribe.return_value = [
        TranscribedWord(
            word="hello",
            start_time_seconds=0.2,
            end_time_seconds=0.8,
            probablility=None,
        )
    ]
    callback = Mock()

    result = TranscriptionWorker(diarizer, whisper).transcribe(
        audio_path, language="en", on_segment=callback
    )

    assert len(result) == 1
    assert result[0].word == "hello"
    assert result[0].speaker_label == "SPEAKER_00"
    assert result[0].start_time_seconds == 1.2
    assert result[0].end_time_seconds == 1.8
    callback.assert_called_once_with(diarizer.diarize.return_value[0])
    whisper.transcribe.assert_called_once()
    assert whisper.transcribe.call_args.kwargs == {"language": "en"}


def test_transcribe_rejects_missing_audio(tmp_path: Path) -> None:
    worker = TranscriptionWorker(Mock(), Mock())

    try:
        worker.transcribe(tmp_path / "missing.wav")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("FileNotFoundError was not raised")
