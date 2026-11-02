from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transcription_worker.models import WhisperModelType
from transcription_worker.whisper import Whisper


def test_full_recording_uses_vad_and_keeps_global_word_timestamps(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.touch()
    whisper = Whisper.__new__(Whisper)
    whisper.model = Mock()
    whisper.model.transcribe.return_value = (
        iter(
            [
                SimpleNamespace(
                    end=42.5,
                    words=[
                        SimpleNamespace(
                            word=" hello", start=42.0, end=42.5, probability=0.9
                        ),
                    ],
                ),
                SimpleNamespace(end=50.0, words=None),
            ]
        ),
        SimpleNamespace(duration=60.0),
    )
    progress = Mock()

    result = whisper.transcribe(audio, language="en", on_progress=progress)

    assert result[0].word == "hello"
    assert result[0].start_time_seconds == 42.0
    assert result[0].probablility == 0.9
    assert [call.args[0] for call in progress.call_args_list] == [42.5, 50.0, 60.0]
    options = whisper.model.transcribe.call_args.kwargs
    assert options["vad_filter"] and options["word_timestamps"]
    assert not options["condition_on_previous_text"]
    assert options["language"] == "en"
    whisper.model.transcribe.assert_called_once()


def test_transcribe_rejects_missing_file(tmp_path: Path):
    whisper = Whisper.__new__(Whisper)
    with pytest.raises(FileNotFoundError):
        whisper.transcribe(tmp_path / "missing.wav")


def test_silence_returns_no_words_and_completes_progress(tmp_path):
    audio = tmp_path / "silence.wav"
    audio.touch()
    whisper = Whisper.__new__(Whisper)
    whisper.model = Mock()
    whisper.model.transcribe.return_value = (iter([]), SimpleNamespace(duration=30.0))
    progress = Mock()

    assert whisper.transcribe(audio, on_progress=progress) == []
    progress.assert_called_once_with(30.0)


@pytest.mark.parametrize(
    "device, resolved, backend, index, precision",
    [
        ("auto", "cpu", "cpu", 0, "int8"),
        ("cuda:2", "cuda:2", "cuda", 2, "float16"),
    ],
)
def test_model_device_selection(
    monkeypatch, device, resolved, backend, index, precision
):
    monkeypatch.setattr(
        "transcription_worker.whisper.resolve_device", Mock(return_value=resolved)
    )
    factory = Mock()
    monkeypatch.setattr("transcription_worker.whisper.WhisperModel", factory)
    Whisper(model_type=WhisperModelType.LARGE, device=device)
    assert factory.call_args.args == ("large-v3",)
    assert factory.call_args.kwargs["device"] == backend
    assert factory.call_args.kwargs["device_index"] == index
    assert factory.call_args.kwargs["compute_type"] == precision


def test_mps_reports_supported_devices():
    with pytest.raises(ValueError, match="does not support MPS"):
        Whisper(device="mps")
