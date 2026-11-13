"""Desktop contract tests: immutable routes, durable jobs, auth and lifecycle."""

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client

from transcription_worker import mcp_server
from transcription_worker.frontend import (
    FrontendService,
    Profile,
    Settings,
    Submission,
    create_app,
)
from transcription_worker.frontend_client import FrontendBackend
from transcription_worker.models import DiarizerModelType, WhisperModelType
from transcription_worker.service import (
    Job,
    JobStatus,
    TranscriptionResult,
    TranscriptionService,
)
from transcription_worker.transcript import format_transcript
from .test_service import service as service, audio_path as audio_path
from .test_process_service import isolated as isolated


class MemorySecrets:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value


@pytest.fixture
def front(tmp_path, monkeypatch):
    monkeypatch.delenv("NOTEIT_SERVER_URL", raising=False)
    monkeypatch.setenv("NOTEIT_OUTPUT_DIR", str(tmp_path / "legacy"))
    service = FrontendService(tmp_path / "desktop", MemorySecrets())
    yield service
    service.close()


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out")


def test_profiles_keep_job_route_and_secret_revision(front, tmp_path):
    front._run = Mock()  # isolate submission snapshot from transport
    audio = tmp_path / "recording.wav"
    audio.touch()
    original = front.save_profile(
        Profile(name="lab", url="http://first:8765", token="original-secret")
    )
    front.save_settings(Settings(target=original["id"]))
    job = front.submit(Submission(audio_path=str(audio)))
    front.save_profile(
        Profile(
            id=original["id"],
            name="changed",
            url="http://second:8765",
            token="new-secret",
        )
    )
    front.save_settings(Settings(target="local"))
    front.delete_profile(original["id"])
    persisted = front.get_job(job["job_id"])
    assert persisted["target"]["url"] == "http://first:8765"
    assert (
        front.credentials.get(persisted["target"]["credential_ref"])
        == "original-secret"
    )
    assert not persisted["diarize"]
    assert "secret" not in (front.directory / "frontend.sqlite3").read_bytes().decode(
        errors="ignore"
    )


def test_desktop_api_auth_validation_and_keychain(front):
    client = TestClient(create_app(front, "desktop-secret"))
    assert client.get("/settings").status_code == 401
    client.headers["Authorization"] = "Bearer desktop-secret"
    assert client.put("/settings", json={"device": "wrong"}).status_code == 400
    assert (
        client.post(
            "/profiles",
            json={"name": "bad", "url": "file:///etc/passwd", "token": "secret"},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/jobs", json={"audio_path": "/missing", "num_speakers": 0}
        ).status_code
        == 422
    )
    assert client.get("/jobs/unknown").status_code == 404
    saved = client.put(
        "/settings", json={"hf_token": "hf-private", "whisper_model": "base"}
    ).json()
    assert saved["whisper_model"] == "base" and "hf_token" not in saved
    assert front.credentials.get("huggingface") == "hf-private"
    assert "hf-private" not in (
        front.directory / "frontend.sqlite3"
    ).read_bytes().decode(errors="ignore")


def test_local_shutdown_terminates_process_and_marks_job(front, tmp_path, monkeypatch):
    audio = tmp_path / "recording.wav"
    audio.touch()
    real_popen = subprocess.Popen
    children = []

    def popen(*args, **kwargs):
        child = real_popen(
            [
                sys.executable,
                "-c",
                "import sys,time; sys.stdin.readline(); time.sleep(120)",
            ],
            **kwargs,
        )
        children.append(child)
        return child

    monkeypatch.setattr("transcription_worker.frontend.subprocess.Popen", popen)
    job = front.submit(Submission(audio_path=str(audio), device="cpu"))
    wait_until(lambda: children and front.process is not None)
    front.close()
    assert children[0].poll() is not None
    assert front.get_job(job["job_id"])["status"] == "interrupted"
    assert audio.exists()  # only private copies may be removed


def test_cancel_local_active_and_queued_jobs(front, tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    children = []

    def popen(*args, **kwargs):
        child = real_popen(
            [
                sys.executable,
                "-c",
                "import sys,time; sys.stdin.readline(); time.sleep(120)",
            ],
            **kwargs,
        )
        children.append(child)
        return child

    monkeypatch.setattr("transcription_worker.frontend.subprocess.Popen", popen)
    audio = tmp_path / "recording.wav"
    audio.touch()
    first = front.submit(Submission(audio_path=str(audio)))
    second = front.submit(Submission(audio_path=str(audio)))
    wait_until(lambda: front.process is not None)
    client = TestClient(create_app(front, "desktop-secret"))
    assert client.post(f"/jobs/{first['job_id']}/cancel").status_code == 401
    client.headers["Authorization"] = "Bearer desktop-secret"
    assert client.post("/jobs/missing/cancel").status_code == 404
    assert (
        client.post(f"/jobs/{second['job_id']}/cancel").json()["status"] == "cancelled"
    )
    assert children[0].poll() is None
    assert client.post(f"/jobs/{first['job_id']}/cancel").status_code == 200
    wait_until(lambda: front.get_job(first["job_id"])["status"] == "cancelled")
    assert children[0].poll() is not None
    assert len(children) == 1 and audio.exists()
    assert front.get_job(first["job_id"])["error"] is None
    front.close()
    restored = FrontendService(front.directory, front.credentials)
    try:
        assert all(job["status"] == "cancelled" for job in restored.list_jobs())
    finally:
        restored.close()


def test_gui_cancel_reaches_remote_process_and_allows_new_job(
    front, isolated, audio_path, tmp_path, monkeypatch
):
    from transcription_worker.backends import RemoteBackend
    from transcription_worker.server import create_app as remote_app

    service, children, _ = isolated
    gate = tmp_path / "remote-cancel-gate"
    monkeypatch.setenv("NOTEIT_TEST_INFERENCE_GATE", str(gate))
    app = remote_app(service, "api-test-token")
    monkeypatch.setattr(
        front,
        "remote",
        lambda target: RemoteBackend(
            "http://testserver", "api-test-token", client=TestClient(app)
        ),
    )
    profile = front.save_profile(
        Profile(name="remote", url="http://testserver", token="api-test-token")
    )
    request = Submission(audio_path=str(audio_path), target=profile["id"])
    job = front.submit(request)
    wait_until(lambda: front.get_job(job["job_id"])["status"] == "transcribing")
    original_remote_id = front.get_job(job["job_id"])["remote_id"]
    front.cancel(job["job_id"])
    wait_until(lambda: front.get_job(job["job_id"])["status"] == "cancelled")
    assert service.wait(original_remote_id, 5).status == JobStatus.CANCELLED
    assert children[0].poll() is not None and service._process is None
    assert audio_path.exists()
    gate.touch()
    new_job = front.submit(request)
    wait_until(lambda: front.get_job(new_job["job_id"])["status"] == "done")
    assert front.get_job(new_job["job_id"])["remote_id"] != original_remote_id


def test_remote_cancel_survives_connection_loss_and_frontend_restart(
    front, tmp_path, monkeypatch
):
    audio = tmp_path / "recording.wav"
    audio.touch()
    profile = front.save_profile(
        Profile(name="remote", url="http://server", token="secret")
    )
    disconnected = threading.Event()
    backend = Mock()
    backend.submit.return_value = {"job_id": "originalremote"}

    def wait(*args):
        disconnected.set()
        raise httpx.ConnectError("offline")

    backend.wait.side_effect = wait
    monkeypatch.setattr(front, "remote", lambda target: backend)
    job = front.submit(Submission(audio_path=str(audio), target=profile["id"]))
    assert disconnected.wait(5)
    front.cancel(job["job_id"])
    assert front.get_job(job["job_id"])["status"] == "cancelling"
    front.close()
    recovered = Mock()
    recovered.wait.return_value = {"job_id": "originalremote", "status": "transcribing"}
    recovered.cancel.return_value = {
        "job_id": "originalremote",
        "status": "cancelled",
        "error": None,
    }
    monkeypatch.setattr(FrontendService, "remote", lambda self, target: recovered)
    restored = FrontendService(front.directory, front.credentials)
    try:
        wait_until(lambda: restored.get_job(job["job_id"])["status"] == "cancelled")
        recovered.submit.assert_not_called()
        recovered.cancel.assert_called_once_with("originalremote")
    finally:
        restored.close()


def test_remote_cancel_is_not_finished_until_server_confirms(
    front, tmp_path, monkeypatch
):
    audio = tmp_path / "recording.wav"
    audio.touch()
    profile = front.save_profile(
        Profile(name="remote", url="http://server", token="secret")
    )
    backend = Mock()
    backend.submit.return_value = {"job_id": "remotejob"}
    backend.wait.return_value = {"job_id": "remotejob", "status": "transcribing"}
    backend.cancel.side_effect = RuntimeError(
        "停止に対応した推論サーバーへ更新してください．"
    )
    monkeypatch.setattr(front, "remote", lambda target: backend)
    job = front.submit(Submission(audio_path=str(audio), target=profile["id"]))
    wait_until(lambda: front.get_job(job["job_id"]).get("remote_id") == "remotejob")
    front.cancel(job["job_id"])
    wait_until(lambda: "更新" in (front.get_job(job["job_id"]).get("error") or ""))
    assert front.get_job(job["job_id"])["status"] == "cancelling"
    backend.cancel.side_effect = None
    backend.cancel.return_value = {
        "job_id": "remotejob",
        "status": "cancelled",
        "error": None,
    }
    wait_until(lambda: front.get_job(job["job_id"])["status"] == "cancelled")


def test_remote_backend_explains_missing_cancel_endpoint():
    from fastapi import FastAPI
    from transcription_worker.backends import RemoteBackend

    client = TestClient(FastAPI())
    backend = RemoteBackend("http://testserver", "secret", client=client)
    with pytest.raises(RuntimeError, match="推論サーバーへ更新"):
        backend.cancel("remotejob")


def test_remote_disconnect_resume_and_result_survive_restart(
    front, tmp_path, monkeypatch
):
    audio = tmp_path / "recording.wav"
    audio.touch()
    profile = front.save_profile(
        Profile(name="lab", url="http://server:8765", token="remote-secret")
    )
    received = threading.Event()
    backend = Mock()
    backend.submit.return_value = {"job_id": "remote123"}

    def disconnected(*args):
        received.set()
        raise httpx.ConnectError("offline")

    backend.wait.side_effect = disconnected
    monkeypatch.setattr(front, "remote", lambda target: backend)
    job = front.submit(Submission(audio_path=str(audio), target=profile["id"]))
    assert received.wait(5)
    wait_until(lambda: front.get_job(job["job_id"])["status"] == "connection_lost")
    front.close()
    assert front.get_job(job["job_id"])["remote_id"] == "remote123"
    audio.unlink()
    completed = Mock()
    completed.wait.return_value = {
        "job_id": "remote123",
        "status": "done",
        "engine": "faster-whisper",
        "device": "cuda",
    }
    completed.get_result.return_value = TranscriptionResult(
        job_id="remote123",
        audio_path="recording.wav",
        language="ja",
        whisper_model=WhisperModelType.SMALL,
        diarizer_model=None,
        elapsed_seconds=1,
        speakers=[],
        utterances=[],
        words=[],
        engine="faster-whisper",
        device="cuda",
    )
    targets = []

    def remote(self, target):
        targets.append(target)
        return completed

    monkeypatch.setattr(FrontendService, "remote", remote)
    restarted = FrontendService(front.directory, front.credentials)
    try:
        wait_until(lambda: restarted.get_job(job["job_id"])["status"] == "done")
        assert (
            restarted.get_result(job["job_id"]).execution_target == "http://server:8765"
        )
        assert restarted.get_result(job["job_id"]).job_id == job["job_id"]
        completed.submit.assert_not_called()
        assert targets[0]["url"] == "http://server:8765"
    finally:
        restarted.close()


@pytest.mark.anyio
async def test_mcp_and_gui_share_settings_jobs_and_plain_transcript(
    front, tmp_path, monkeypatch
):
    client = TestClient(create_app(front, "desktop-secret"))
    client.headers["Authorization"] = "Bearer desktop-secret"
    backend = FrontendBackend.__new__(FrontendBackend)
    backend.client = client
    monkeypatch.setattr(mcp_server, "_backend", backend)
    front._run = Mock()
    audio = tmp_path / "recording.wav"
    audio.touch()
    client.put(
        "/settings", json={"whisper_model": "base", "language": "en", "device": "cpu"}
    )
    async with Client(mcp_server.server) as mcp:
        started = await mcp.call_tool("start_transcription", {"audio_path": str(audio)})
        assert not started.is_error, started.content
        job_id = started.structured_content["job_id"]
        job = client.get(f"/jobs/{job_id}").json()
        assert job["whisper_model"] == "base"
        assert job["options"]["device"] == "cpu" and job["options"]["diarize"] is False
        assert job["options"]["language"] == "en"
        result = TranscriptionResult(
            job_id=job_id,
            audio_path=str(audio),
            language="en",
            whisper_model=WhisperModelType.BASE,
            diarizer_model=None,
            elapsed_seconds=1,
            speakers=[],
            utterances=[
                {
                    "speaker_label": None,
                    "text": "Hello",
                    "start_time_seconds": 5,
                    "end_time_seconds": 6,
                }
            ],
            words=[],
        )
        job["status"] = "done"
        front._write_job(job, result)
        text = await mcp.call_tool("get_transcript", {"job_id": job_id})
        assert "[00:00:05] Hello" in text.content[0].text
        assert (
            "None:" not in text.content[0].text
            and "pyannote-" not in text.content[0].text
        )
        assert client.get(f"/jobs/{job_id}/text").json()["text"] == "[00:00:05] Hello"
        updated = await mcp.call_tool(
            "update_transcription_settings", {"whisper_model": "tiny"}
        )
        assert not updated.is_error
        assert client.get("/settings").json()["whisper_model"] == "tiny"


def test_server_restart_marks_unfinished_jobs_interrupted(service, audio_path):
    svc, _, _ = service
    pending = Job(
        id="pending123",
        audio_path=audio_path,
        language="ja",
        whisper_model=WhisperModelType.SMALL,
        diarizer_model=None,
        speaker_hints={},
        status=JobStatus.TRANSCRIBING,
    )
    svc._persist(pending)
    restarted = TranscriptionService(svc.config)
    assert restarted.wait(pending.id, 0).status == JobStatus.INTERRUPTED


def test_legacy_result_without_engine_is_readable():
    result = TranscriptionResult.model_validate(
        dict(
            job_id="old123",
            audio_path="old.wav",
            language="ja",
            whisper_model="small",
            diarizer_model="community",
            elapsed_seconds=1,
            speakers=["SPEAKER_00"],
            utterances=[
                dict(
                    speaker_label="SPEAKER_00",
                    text="旧結果",
                    start_time_seconds=1,
                    end_time_seconds=2,
                )
            ],
            words=[],
        )
    )
    assert result.engine is None
    assert format_transcript(result.utterances) == "[00:00:01] SPEAKER_00: 旧結果"


def test_concurrent_clients_reuse_one_detached_service(tmp_path, monkeypatch):
    directory = tmp_path / "singleton"
    env = {
        **os.environ,
        "NOTEIT_DATA_DIR": str(directory),
        "NOTEIT_OUTPUT_DIR": str(directory / "legacy"),
    }
    env.pop("NOTEIT_SERVER_URL", None)
    command = [
        sys.executable,
        "-c",
        "import json; from transcription_worker.frontend import ensure_service; print(json.dumps(ensure_service()))",
    ]

    def launch(_):
        output = subprocess.check_output(command, env=env, text=True, timeout=30)
        return json.loads(output)

    endpoint = None
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = list(pool.map(launch, range(2)))
        endpoint = first
        assert first == second
        assert (directory / "endpoint.json").stat().st_mode & 0o777 == 0o600
        assert httpx.get(first["url"] + "/settings", timeout=2).status_code == 401
        assert (
            httpx.get(
                first["url"] + "/settings",
                headers={"Authorization": "Bearer " + first["token"]},
                timeout=2,
            ).json()["whisper_model"]
            == "small"
        )
    finally:
        if endpoint is None and (directory / "endpoint.json").exists():
            endpoint = json.loads((directory / "endpoint.json").read_text())
        if endpoint:
            httpx.post(
                endpoint["url"] + "/shutdown",
                headers={"Authorization": "Bearer " + endpoint["token"]},
                timeout=10,
            )
            wait_until(lambda: not (directory / "endpoint.json").exists())


def test_imports_previous_saved_results_into_shared_history(front, tmp_path):
    result = TranscriptionResult(
        job_id="legacy123",
        audio_path="old.wav",
        language="ja",
        whisper_model=WhisperModelType.SMALL,
        diarizer_model=DiarizerModelType.COMMUNITY,
        elapsed_seconds=1,
        speakers=[],
        utterances=[],
        words=[],
    )
    front.config.output_dir.mkdir()
    (front.config.output_dir / "legacy123.json").write_text(result.model_dump_json())
    restarted = FrontendService(front.directory, front.credentials)
    try:
        assert restarted.get_job("legacy123")["status"] == "done"
        assert restarted.get_result("legacy123") == result
    finally:
        restarted.close()
