"""Exercise real process exit/termination without downloading model weights."""

import json
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import pytest

from transcription_worker.models import DiarizerModelType, WhisperModelType
from transcription_worker.process_service import ProcessTranscriptionService
from transcription_worker.server import create_app
from transcription_worker.service import JobStatus, ServiceConfig


STUB = r"""
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
from transcription_worker.service import TranscriptionService
from transcription_worker.schema import SpeakerAssignedWord

def load(service, whisper_model, diarizer_model):
    assert service.config.hf_token == "private-test-token"
    assert service.config.diarizer_device == "cuda"
    assert service.config.compile_whisper is True
    class Worker:
        _whisper = SimpleNamespace(engine="test-engine", device=service.config.device)
        def transcribe(self, audio_path, *, language, on_diarized, on_segment, **hints):
            print("inference diagnostics go to stderr", flush=True)
            with open(os.environ["NOTEIT_TEST_TRACE"], "a") as trace:
                trace.write(json.dumps({"pid":os.getpid(), "whisper":whisper_model,
                    "diarizer":diarizer_model, "language":language, "hints":hints})+"\n")
            on_diarized([])
            if os.environ.get("NOTEIT_TEST_INFERENCE_GATE"):
                while not Path(os.environ["NOTEIT_TEST_INFERENCE_GATE"]).exists():
                    time.sleep(0.02)
            return [SpeakerAssignedWord(word="hello", start_time_seconds=0,
                end_time_seconds=1, probablility=None,
                speaker_label="SPEAKER_00" if diarizer_model else None)]
    return Worker()

TranscriptionService._load_worker = load
from transcription_worker.inference_process import main
main()
if os.environ.get("NOTEIT_TEST_EXIT_GATE"):
    Path(os.environ["NOTEIT_TEST_RESULT_SENT"]).touch()
    while not Path(os.environ["NOTEIT_TEST_EXIT_GATE"]).exists():
        time.sleep(0.02)
"""


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTEIT_TEMP_DIR", str(tmp_path / "temporary"))
    monkeypatch.setenv("NOTEIT_TEST_TRACE", str(tmp_path / "trace.jsonl"))
    config = ServiceConfig(
        hf_token="private-test-token",
        whisper_model=WhisperModelType.SMALL,
        diarizer_model=DiarizerModelType.COMMUNITY,
        device="cuda",
        diarizer_device="cuda",
        compile_whisper=True,
        output_dir=tmp_path / "transcripts",
    )
    service = ProcessTranscriptionService(config)
    children = []
    real_popen = subprocess.Popen

    def launch(script=STUB):
        def popen(command, **kwargs):
            if command != [
                sys.executable,
                "-m",
                "transcription_worker.inference_process",
            ]:
                return real_popen(command, **kwargs)
            assert kwargs["start_new_session"] is True
            child = real_popen([sys.executable, "-c", script], **kwargs)
            children.append(child)
            return child

        monkeypatch.setattr(
            "transcription_worker.process_service.subprocess.Popen", popen
        )

    launch()
    try:
        yield service, children, launch
    finally:
        service.close()


def test_http_jobs_cold_start_and_preserve_progress_results_and_cache(
    isolated, tmp_path, monkeypatch
):
    service, children, _ = isolated
    cache = tmp_path / "model-cache"
    cache.mkdir()
    weights = cache / "cached-model"
    weights.write_text("existing weights")
    monkeypatch.setenv("HF_HOME", str(cache))
    libraries = {
        name
        for name in ("torch", "ctranslate2", "faster_whisper")
        if name in sys.modules
    }

    def unexpected_load(*args):
        raise AssertionError("The HTTP parent must not load models")

    monkeypatch.setattr(service, "_load_worker", unexpected_load)
    with TestClient(create_app(service, "api-test-token")) as client:
        headers = {"Authorization": "Bearer api-test-token"}
        assert client.get("/models").status_code == 401
        assert client.get("/models", headers=headers).status_code == 200
        ids = []
        for diarize in (False, True):
            response = client.post(
                "/jobs",
                headers=headers,
                files={"audio": ("meeting.wav", b"test audio")},
                data={
                    "language": "en",
                    "whisper_model": "tiny",
                    "diarize": str(diarize).lower(),
                    "num_speakers": "2",
                    "source_name": "meeting.wav",
                },
            )
            assert response.status_code == 200, response.text
            ids.append(response.json()["job_id"])
        for job_id in ids:
            job = service.wait(job_id, 10)
            assert job.status == JobStatus.DONE, job.error
            assert job.engine == "test-engine" and job.device == "cuda"
            response = client.get(f"/jobs/{job_id}/result", headers=headers)
            assert response.status_code == 200
            assert response.json()["audio_path"] == "meeting.wav"
        assert all(child.poll() == 0 for child in children)
        assert service._process is None
        assert service._whisper is None and service._diarizer is None
    traces = [
        json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()
    ]
    assert len({trace["pid"] for trace in traces}) == 2
    assert [trace["diarizer"] for trace in traces] == [None, "community"]
    assert all(
        trace["whisper"] == "tiny" and trace["language"] == "en" for trace in traces
    )
    assert all(trace["hints"] == {"num_speakers": 2} for trace in traces)
    assert service.get_result(ids[0]).speakers == []
    assert service.get_result(ids[1]).speakers == ["SPEAKER_00"]
    assert weights.read_text() == "existing weights"
    assert not list((tmp_path / "temporary").iterdir())
    assert not list((tmp_path / "uploads").iterdir())
    assert libraries == {
        name
        for name in ("torch", "ctranslate2", "faster_whisper")
        if name in sys.modules
    }
    restarted = ProcessTranscriptionService(service.config)
    try:
        assert restarted.get_result(ids[0]) == service.get_result(ids[0])
    finally:
        restarted.close()


def test_completion_waits_for_process_exit_before_starting_next_job(
    isolated, tmp_path, monkeypatch
):
    service, children, _ = isolated
    gate = tmp_path / "exit-gate"
    sent = tmp_path / "result-sent"
    monkeypatch.setenv("NOTEIT_TEST_EXIT_GATE", str(gate))
    monkeypatch.setenv("NOTEIT_TEST_RESULT_SENT", str(sent))
    audio = tmp_path / "original.wav"
    audio.touch()
    first = service.submit(audio, diarize=False)
    second = service.submit(audio, diarize=False)
    wait_until(sent.exists)
    assert not first.finished.is_set()
    assert first.result is None and first.status != JobStatus.DONE
    assert second.status == JobStatus.QUEUED
    assert len(children) == 1 and children[0].poll() is None
    gate.touch()
    assert service.wait(first.id, 10).status == JobStatus.DONE
    assert service.wait(second.id, 10).status == JobStatus.DONE
    assert len(children) == 2 and all(child.poll() == 0 for child in children)
    assert audio.exists()


def test_http_remains_responsive_and_reports_progress_during_inference(
    isolated, tmp_path, monkeypatch
):
    service, children, _ = isolated
    gate = tmp_path / "inference-gate"
    monkeypatch.setenv("NOTEIT_TEST_INFERENCE_GATE", str(gate))
    audio = tmp_path / "original.wav"
    audio.touch()
    with TestClient(create_app(service, "api-test-token")) as client:
        job = service.submit(audio, diarize=False)
        wait_until(lambda: job.status == JobStatus.TRANSCRIBING)
        headers = {"Authorization": "Bearer api-test-token"}
        assert client.get("/health").status_code == 200
        assert client.get("/models", headers=headers).status_code == 200
        snapshot = client.get(f"/jobs/{job.id}", headers=headers).json()
        assert snapshot["status"] == "transcribing"
        assert snapshot["engine"] == "test-engine" and snapshot["device"] == "cuda"
        assert client.get(f"/jobs/{job.id}/result", headers=headers).status_code == 409
        assert children[0].poll() is None
        gate.touch()
        assert service.wait(job.id, 10).status == JobStatus.DONE


@pytest.mark.parametrize(
    "script,expected",
    [
        ("import sys; sys.stdin.readline(); sys.exit(7)", "exited (7)"),
        ("import sys; sys.stdin.readline()", "without a valid result"),
        (
            "import sys,time; sys.stdin.readline(); print('invalid',flush=True); time.sleep(120)",
            "JSONDecodeError",
        ),
        (
            "import sys,json; sys.stdin.readline(); print(json.dumps({'status':'failed','error':'model load failed'}))",
            "model load failed",
        ),
    ],
)
def test_failed_child_is_reaped_and_private_audio_is_removed(
    isolated, tmp_path, script, expected
):
    service, children, launch = isolated
    launch(script)
    audio = tmp_path / "owned.upload"
    audio.touch()
    job = service.submit(audio, diarize=False, delete_audio_after=True)
    service.wait(job.id, 10)
    assert job.status == JobStatus.FAILED, job.snapshot()
    assert expected in job.error
    assert children[0].poll() is not None
    assert service._process is None
    assert not audio.exists()
    assert not list((tmp_path / "temporary").iterdir())


def test_queue_recovers_after_child_crash(isolated, tmp_path):
    service, children, launch = isolated
    audio = tmp_path / "original.wav"
    audio.touch()
    launch("import sys; sys.stdin.readline(); sys.exit(7)")
    failed = service.submit(audio, diarize=False)
    assert service.wait(failed.id, 10).status == JobStatus.FAILED
    launch()
    recovered = service.submit(audio, diarize=False)
    assert service.wait(recovered.id, 10).status == JobStatus.DONE
    assert len(children) == 2 and all(child.poll() is not None for child in children)
    assert audio.exists()


def test_http_shutdown_kills_stuck_child_and_interrupts_queued_uploads(
    isolated, tmp_path, monkeypatch
):
    service, children, launch = isolated
    ready = tmp_path / "ready"
    monkeypatch.setenv("NOTEIT_TEST_READY", str(ready))
    monkeypatch.setattr(
        "transcription_worker.process_service.PROCESS_STOP_TIMEOUT", 0.1
    )
    launch(
        "import os,sys,time,signal; from pathlib import Path; sys.stdin.readline(); "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "Path(os.environ['NOTEIT_TEST_READY']).touch(); time.sleep(120)"
    )
    first = tmp_path / "first.upload"
    second = tmp_path / "second.upload"
    first.touch()
    second.touch()
    with TestClient(create_app(service, "api-test-token")):
        active = service.submit(first, diarize=False, delete_audio_after=True)
        queued = service.submit(second, diarize=False, delete_audio_after=True)
        wait_until(ready.exists)
    assert children[0].poll() is not None
    assert len(children) == 1 and service._process is None
    assert active.status == queued.status == JobStatus.INTERRUPTED
    assert active.finished.is_set() and queued.finished.is_set()
    assert not first.exists() and not second.exists()
    assert not list((tmp_path / "temporary").iterdir())
    with pytest.raises(RuntimeError, match="shutting down"):
        service.submit(first)


def test_server_entrypoint_uses_process_service(monkeypatch):
    from unittest.mock import Mock
    from transcription_worker import server

    factory = Mock()
    app = Mock(return_value="test-app")
    run = Mock()
    monkeypatch.setenv("NOTEIT_API_TOKEN", "api-test-token")
    monkeypatch.setenv("NOTEIT_WHISPER_MODEL", "tiny")
    monkeypatch.setattr(sys, "argv", ["noteit-server"])
    monkeypatch.setattr(server, "ProcessTranscriptionService", factory)
    monkeypatch.setattr(server, "create_app", app)
    monkeypatch.setattr("uvicorn.run", run)
    server.main()
    factory.assert_called_once()
    app.assert_called_once_with(factory.return_value, "api-test-token")
    assert run.call_args.args == ("test-app",)
    assert run.call_args.kwargs["timeout_graceful_shutdown"] == 5


def test_cancel_active_and_queued_jobs_without_stopping_the_service(
    isolated, tmp_path, monkeypatch
):
    service, children, _ = isolated
    gate = tmp_path / "cancel-gate"
    monkeypatch.setenv("NOTEIT_TEST_INFERENCE_GATE", str(gate))
    audio = tmp_path / "original.wav"
    audio.touch()
    first = service.submit(audio, diarize=False, job_id="first")
    queued = service.submit(audio, diarize=False, job_id="queued")
    wait_until(lambda: first.status == JobStatus.TRANSCRIBING)
    with TestClient(create_app(service, "api-test-token")) as client:
        headers = {"Authorization": "Bearer api-test-token"}
        assert client.post("/jobs/first/cancel").status_code == 401
        assert client.post("/jobs/unknown/cancel", headers=headers).status_code == 404
        assert (
            client.post("/jobs/queued/cancel", headers=headers).json()["status"]
            == "cancelled"
        )
        assert queued.finished.is_set() and children[0].poll() is None
        response = client.post("/jobs/first/cancel", headers=headers)
        assert response.status_code == 200
        assert service.wait(first.id, 10).status == JobStatus.CANCELLED
        assert children[0].poll() is not None and service._process is None
        assert first.error is None and first.result is None and audio.exists()
        assert client.get("/jobs/first/result", headers=headers).status_code == 409
        gate.touch()
        restarted_job = service.submit(audio, diarize=False, job_id="first")
        assert service.wait(restarted_job.id, 10).status == JobStatus.DONE
        assert len(children) == 2
        assert (
            client.post("/jobs/first/cancel", headers=headers).json()["status"]
            == "done"
        )
    restored = ProcessTranscriptionService(service.config)
    try:
        assert restored.get_job("queued").status == JobStatus.CANCELLED
    finally:
        restored.close()
