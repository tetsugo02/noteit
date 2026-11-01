from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile
from transcription_worker.whisper import Whisper


def make_whisper(raw_result: object) -> Whisper:
    whisper = Whisper.__new__(Whisper)
    whisper.pipe = Mock(return_value=raw_result)
    whisper.is_compiled = False
    return whisper


def test_transcribe_returns_current_worker_schema(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    soundfile.write(audio_path, np.zeros((1600, 2), dtype="float32"), 16_000)
    whisper = make_whisper(
        {
            "text": " hello world",
            "chunks": [
                {"text": " hello", "timestamp": (0.0, 0.5)},
                {"text": " world", "timestamp": (0.5, 1.0)},
            ],
        }
    )

    result = whisper.transcribe(audio_path, language="en")

    assert [word.word for word in result] == ["hello", "world"]
    assert result[0].start_time_seconds == 0.0
    assert result[1].end_time_seconds == 1.0
    assert result[0].probablility is None
    whisper.pipe.assert_called_once()  # type: ignore
    audio = whisper.pipe.call_args.args[0]  # type: ignore
    assert audio["sampling_rate"] == 16_000
    assert audio["raw"].shape == (1600,)  # stereo downmixed to mono
    assert whisper.pipe.call_args.kwargs == dict(  # type: ignore
        return_timestamps="word",
        decoder_kwargs={"clean_up_tokenization_spaces": False},
        language="en",
    )


def test_transcribe_rejects_missing_file(tmp_path: Path) -> None:
    whisper = make_whisper({"chunks": []})

    with pytest.raises(FileNotFoundError):
        whisper.transcribe(tmp_path / "missing.wav")


def test_transcribe_rejects_result_without_word_timestamps(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    soundfile.write(audio_path, np.zeros(1600, dtype="float32"), 16_000)
    whisper = make_whisper({"text": "hello"})

    with pytest.raises(ValueError, match="word timestamps"):
        whisper.transcribe(audio_path)
