from pathlib import Path
from unittest.mock import Mock

import pytest
from pydub import AudioSegment
from transcription_worker.schema import SpeakerSegment, TranscribedWord
from transcription_worker.worker import TranscriptionWorker, assign_speakers


def word(start, end, text="hello"):
    return TranscribedWord(
        word=text, start_time_seconds=start, end_time_seconds=end, probablility=0.8
    )


def turn(start, end, speaker):
    return SpeakerSegment(start_time=start, end_time=end, speaker_label=speaker)


def test_transcribe_recognizes_once_across_multiple_turns(tmp_path):
    audio = tmp_path / "audio.wav"
    AudioSegment.silent(duration=3000).export(audio, format="wav")
    diarizer = Mock()
    diarizer.diarize.return_value = [turn(1, 2, "A"), turn(2, 3, "B")]
    whisper = Mock()

    def recognize(path, *, language, on_progress):
        assert Path(path).is_file()
        assert language == "en"
        on_progress(1.5)
        callback.assert_not_called()
        on_progress(2.5)
        callback.assert_called_once_with(diarizer.diarize.return_value[0])
        return [word(1.2, 1.8), word(2.1, 2.8)]

    whisper.transcribe.side_effect = recognize
    callback = Mock()
    result = TranscriptionWorker(diarizer, whisper).transcribe(
        audio, language="en", num_speakers=2, on_segment=callback
    )

    assert [w.speaker_label for w in result] == ["A", "B"]
    assert result[0].start_time_seconds == 1.2
    assert result[1].end_time_seconds == 2.8
    assert callback.call_count == 2
    whisper.transcribe.assert_called_once()
    assert diarizer.diarize.call_args.kwargs == {"num_speakers": 2}
    assert not whisper.transcribe.call_args.args[0].exists()  # temporary WAV cleaned up


@pytest.mark.parametrize(
    "words, turns, labels",
    [
        ([word(0.8, 1.5)], [turn(0, 1, "A"), turn(1, 2, "B")], ["B"]),
        (
            [word(0, 1)],
            [turn(0, 0.3, "A"), turn(0.4, 0.7, "A"), turn(0, 0.5, "B")],
            ["A"],
        ),
        ([word(0.5, 1.5)], [turn(0, 1, "A"), turn(1, 2, "B")], ["A"]),
        ([word(3, 4)], [turn(0, 1, "A")], ["SPEAKER_UNKNOWN"]),
        ([word(1, 1)], [turn(0, 1, "A"), turn(1, 2, "B")], ["B"]),
        ([word(0, 1)], [], ["SPEAKER_UNKNOWN"]),
        ([], [turn(0, 1, "A")], []),
        ([word(2, 3), word(0, 1)], [turn(2, 3, "B"), turn(0, 1, "A")], ["A", "B"]),
    ],
)
def test_assignment_handles_boundaries_overlap_silence_and_order(words, turns, labels):
    result = assign_speakers(words, turns)
    assert [w.speaker_label for w in result] == labels
    assert all(w.probablility == 0.8 for w in result)


def test_transcribe_rejects_missing_audio(tmp_path: Path):
    worker = TranscriptionWorker(Mock(), Mock())
    with pytest.raises(FileNotFoundError):
        worker.transcribe(tmp_path / "missing.wav")


def test_whisper_only_returns_no_speaker_and_preserves_global_times(tmp_path):
    audio = tmp_path / "audio.wav"
    AudioSegment.silent(duration=3000).export(audio, format="wav")
    whisper = Mock()
    whisper.transcribe.return_value = [word(2.1, 2.8)]
    result = TranscriptionWorker(None, whisper).transcribe(audio)
    assert result[0].speaker_label is None
    assert result[0].start_time_seconds == 2.1
    whisper.transcribe.assert_called_once()
