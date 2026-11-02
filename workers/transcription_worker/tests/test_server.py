from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydub import AudioSegment
from transcription_worker.backends import RemoteBackend
from transcription_worker.models import DiarizerModelType, WhisperModelType
from transcription_worker.server import create_app
from transcription_worker.service import ServiceConfig, TranscriptionService

from .test_service import FakeWorker

TOKEN = "secret-token"


@pytest.fixture
def setup(tmp_path: Path):
    config = ServiceConfig(
        hf_token="hf",
        whisper_model=WhisperModelType.TURBO,
        diarizer_model=DiarizerModelType.COMMUNITY,
        device="cuda",
        diarizer_device="cuda",
        compile_whisper=False,
        output_dir=tmp_path / "server" / "transcripts",
    )
    service = TranscriptionService(config)
    worker = FakeWorker()
    seen_audio: list[tuple[Path, int, int]] = []

    def load_worker(*_):
        original = worker.transcribe

        def transcribe(audio_path, **kwargs):
            audio = AudioSegment.from_file(audio_path)
            seen_audio.append((audio_path, audio.frame_rate, audio.channels))
            return original(audio_path, **kwargs)

        worker.transcribe = transcribe
        return worker

    service._load_worker = Mock(side_effect=load_worker)  # type: ignore[method-assign]
    client = TestClient(create_app(service, TOKEN))

    audio = tmp_path / "meeting.m4a"
    AudioSegment.silent(duration=1500, frame_rate=48000).set_channels(2).export(
        audio, format="ipod"
    )
    return service, client, audio, seen_audio


def test_requires_token(setup) -> None:
    _, client, _, _ = setup
    assert client.get("/health").status_code == 200
    assert client.get("/models").status_code == 401
    assert (
        client.get("/models", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )


def test_remote_backend_round_trip(setup) -> None:
    service, client, audio, seen_audio = setup
    backend = RemoteBackend("http://testserver", TOKEN, client=client)

    assert backend.describe()["defaults"]["whisper_model"] == "turbo"

    snapshot = backend.submit(
        audio, whisper_model=WhisperModelType.SMALL, num_speakers=2
    )
    job_id = snapshot["job_id"]
    snapshot = backend.wait(job_id, 10)
    assert snapshot["status"] == "done", snapshot
    assert snapshot["audio_path"] == "meeting.m4a"

    # Uploaded audio was downmixed to mono Opus, then deleted after use.
    # (ffmpeg always decodes Opus at 48 kHz; the worker resamples to 16 kHz.)
    uploaded_path, _frame_rate, channels = seen_audio[0]
    assert channels == 1
    assert not uploaded_path.exists()

    result = backend.get_result(job_id)
    assert [u.text for u in result.utterances] == ["よろしくお願いします", "はい"]
    assert result.whisper_model is WhisperModelType.SMALL

    # Same audio + options: no second upload or second run.
    assert (
        backend.submit(audio, whisper_model=WhisperModelType.SMALL, num_speakers=2)[
            "job_id"
        ]
        == job_id
    )
    assert service._load_worker.call_count == 1

    # After a server restart the saved result is still served.
    restarted = TestClient(create_app(TranscriptionService(service.config), TOKEN))
    again = RemoteBackend("http://testserver", TOKEN, client=restarted)
    assert again.wait(job_id, 0)["status"] == "done"
    assert again.get_result(job_id).utterances == result.utterances


def test_unknown_job_is_key_error(setup) -> None:
    _, client, _, _ = setup
    backend = RemoteBackend("http://testserver", TOKEN, client=client)
    with pytest.raises(KeyError):
        backend.wait("doesnotexist", 0)
    with pytest.raises(KeyError):
        backend.get_result("..")


def test_remote_submission_does_not_reuse_legacy_pipeline_result(setup):
    import hashlib
    import json

    service, client, audio, _ = setup
    options = {"language": "ja", "whisper_model": "small", "num_speakers": 2}
    digest = hashlib.sha256(audio.read_bytes())
    digest.update(json.dumps(options, sort_keys=True).encode())
    legacy_id = digest.hexdigest()[:16]
    service.submit(
        audio, job_id=legacy_id, whisper_model=WhisperModelType.SMALL, num_speakers=2
    )
    assert service.wait(legacy_id, 10).status.value == "done"

    backend = RemoteBackend("http://testserver", TOKEN, client=client)
    current = backend.submit(
        audio, whisper_model=WhisperModelType.SMALL, num_speakers=2
    )
    assert current["job_id"] != legacy_id
    assert backend.wait(current["job_id"], 10)["status"] == "done"
    assert service._load_worker.call_count == 2


def test_upload_through_symlink_is_not_deleted_before_inference(setup, tmp_path):
    from dataclasses import replace
    from threading import Event

    service, _, audio, _ = setup
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "server", target_is_directory=True)
    service = TranscriptionService(
        replace(service.config, output_dir=alias / "transcripts")
    )
    release = Event()

    def load(*_):
        assert release.wait(5)
        return FakeWorker()

    service._load_worker = load
    client = TestClient(create_app(service, TOKEN))
    try:
        with audio.open("rb") as file:
            response = client.post(
                "/jobs",
                headers={"Authorization": f"Bearer {TOKEN}"},
                files={"audio": ("audio.m4a", file)},
            )
        assert response.status_code == 200
        job = response.json()
        assert list((alias / "uploads").glob("*.upload"))
    finally:
        release.set()
    assert service.wait(job["job_id"], 10).status.value == "done"
