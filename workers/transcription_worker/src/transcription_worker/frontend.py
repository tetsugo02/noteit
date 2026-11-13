"""Single-user desktop service. No inference libraries are imported at startup."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .backends import FINISHED, RemoteBackend, model_catalog
from .models import DEVICE_CHOICES, DiarizerModelType, WhisperModelType
from .process_service import _stop_process
from .service import ServiceConfig, TranscriptionResult
from .transcript import format_transcript


def data_directory() -> Path:
    default = (
        Path.home() / "Library/Application Support/noteit"
        if sys.platform == "darwin"
        else Path.home() / ".local/share/noteit"
    )
    directory = Path(os.environ.get("NOTEIT_DATA_DIR", default)).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    return directory


class KeychainSecrets:
    """Use only the macOS Keychain backend; never silently fall back to a file."""

    def _backend(self):
        if sys.platform != "darwin":
            raise RuntimeError("Desktop credentials require macOS Keychain.")
        from keyring.backends.macOS import Keyring

        return Keyring()

    def get(self, key: str) -> str | None:
        try:
            return self._backend().get_password("noteit", key)
        except Exception as error:
            raise RuntimeError("Keychainの認証情報を読み込めません．") from error

    def set(self, key: str, value: str):
        try:
            self._backend().set_password("noteit", key, value)
        except Exception as error:
            raise RuntimeError("Keychainに認証情報を保存できません．") from error


class Submission(BaseModel):
    audio_path: str
    target: str | None = None
    whisper_model: WhisperModelType = WhisperModelType.SMALL
    language: str | None = "ja"
    diarize: bool = False
    diarizer_model: DiarizerModelType = DiarizerModelType.COMMUNITY
    num_speakers: int | None = Field(None, ge=1)
    min_speakers: int | None = Field(None, ge=1)
    max_speakers: int | None = Field(None, ge=1)
    device: str = "auto"


class Profile(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1, max_length=100)
    url: str
    token: str | None = None


class Settings(BaseModel):
    target: str = "local"
    device: str = "auto"
    whisper_model: WhisperModelType = WhisperModelType.SMALL
    language: str | None = "ja"
    diarize: bool = False
    hf_token: str | None = None


class FrontendService:
    def __init__(self, directory: Path, credentials=None):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.credentials = credentials or KeychainSecrets()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="noteit-front"
        )
        self.stopping = threading.Event()
        self.process_lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.process_job_id: str | None = None
        self.job_lock = threading.RLock()
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL, result TEXT);
            """)
        self.config = ServiceConfig.from_env(default_model=WhisperModelType.SMALL)
        with self.db() as db:
            initialized = db.execute("SELECT 1 FROM settings").fetchone()
        if not initialized:
            settings = Settings()
            if os.environ.get("NOTEIT_SERVER_URL"):
                profile = self.save_profile(
                    Profile(
                        name="環境変数のサーバー",
                        url=os.environ["NOTEIT_SERVER_URL"],
                        token=os.environ.get("NOTEIT_API_TOKEN"),
                    )
                )
                settings.target = profile["id"]
            self.save_settings(settings)
        # Keep transcripts from the pre-desktop CLI/MCP accessible by their IDs.
        for path in self.config.output_dir.glob("*.json"):
            try:
                result = TranscriptionResult.model_validate_json(path.read_text())
                with self.db() as db:
                    exists = db.execute(
                        "SELECT 1 FROM jobs WHERE id=?", (result.job_id,)
                    ).fetchone()
                if not exists:
                    job = dict(
                        job_id=result.job_id,
                        status="done",
                        audio_path=result.audio_path,
                        target={"type": "imported", "name": "以前の結果"},
                        options={},
                        whisper_model=result.whisper_model.value,
                        diarizer_model=result.diarizer_model.value
                        if result.diarizer_model
                        else None,
                        diarize=result.diarizer_model is not None,
                        engine=result.engine,
                        device=result.device,
                        elapsed_seconds=result.elapsed_seconds,
                        segments_done=0,
                        segments_total=None,
                        error=None,
                        created_at=path.stat().st_mtime,
                        started_at=None,
                    )
                    self._write_job(job, result)
            except (ValueError, OSError):
                continue
        # Remote jobs can be polled after restart without their input file.
        for job in self.list_jobs():
            if job["status"] not in FINISHED:
                if job["target"]["type"] == "remote" and job.get("remote_id"):
                    self.executor.submit(self._run, job["job_id"], True)
                else:
                    job.update(
                        status="interrupted",
                        error="サービス終了により中断しました．手動で再実行してください．",
                    )
                    self._write_job(job)

    def db(self):
        return sqlite3.connect(self.directory / "frontend.sqlite3", timeout=30)

    def settings(self):
        with self.db() as db:
            row = db.execute("SELECT data FROM settings WHERE id=1").fetchone()
        return (
            json.loads(row[0])
            if row
            else Settings().model_dump(mode="json", exclude={"hf_token"})
        )

    def save_settings(self, settings: Settings):
        if settings.device not in DEVICE_CHOICES:
            raise ValueError("Unknown device")
        if settings.target != "local":
            self.profile(settings.target)
        if settings.hf_token is not None:
            self.credentials.set("huggingface", settings.hf_token)
        data = settings.model_dump(mode="json", exclude={"hf_token"})
        with self.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO settings VALUES (1, ?)", (json.dumps(data),)
            )
        return data

    def profiles(self):
        with self.db() as db:
            return [
                json.loads(row[0]) for row in db.execute("SELECT data FROM profiles")
            ]

    def profile(self, profile_id):
        return next(
            (p for p in self.profiles() if p["id"] == profile_id), None
        ) or self._missing("Unknown profile")

    @staticmethod
    def _missing(message):
        raise KeyError(message)

    def save_profile(self, profile: Profile):
        url = urlsplit(profile.url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "HTTP(S) URL without credentials, query or fragment is required"
            )
        existing = self.profile(profile.id) if profile.id else None
        token = profile.token or (
            self.credentials.get(existing["credential_ref"]) if existing else None
        )
        if not token:
            raise ValueError("API token is required")
        # Each revision gets an immutable credential reference. Existing jobs
        # keep the URL and credential that were selected at submission time.
        credential_ref = "remote-" + uuid.uuid4().hex
        self.credentials.set(credential_ref, token)
        data = dict(
            id=profile.id or uuid.uuid4().hex,
            name=profile.name,
            url=profile.url.rstrip("/"),
            credential_ref=credential_ref,
        )
        with self.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO profiles VALUES (?, ?)",
                (data["id"], json.dumps(data)),
            )
        return data

    def delete_profile(self, profile_id):
        self.profile(profile_id)
        with self.db() as db:
            db.execute("DELETE FROM profiles WHERE id=?", (profile_id,))
        if self.settings()["target"] == profile_id:
            self.save_settings(Settings(**{**self.settings(), "target": "local"}))

    def remote(self, target):
        token = self.credentials.get(target["credential_ref"])
        if not token:
            raise RuntimeError("保存されたAPIトークンをKeychainから取得できません．")
        return RemoteBackend(target["url"], token)

    def describe(self, target=None):
        target = target or self.settings()["target"]
        if target == "local":
            catalog = model_catalog(
                WhisperModelType(self.settings()["whisper_model"]),
                self.config.diarizer_model,
            )
            catalog["defaults"]["diarize"] = False
            catalog["backend"] = {"type": "local"}
            return catalog
        backend = self.remote(self.profile(target))
        try:
            return backend.describe()
        finally:
            backend.client.close()

    def _write_job(self, job, result: TranscriptionResult | None = None):
        with self.job_lock, self.db() as db:
            previous = db.execute(
                "SELECT data FROM jobs WHERE id=?", (job["job_id"],)
            ).fetchone()
            if previous:
                previous = json.loads(previous[0])
                if previous.get("cancel_requested"):
                    job["cancel_requested"] = True
                    if previous["status"] == "cancelled":
                        job.update(previous)
                        result = None
                    elif job["status"] not in FINISHED:
                        job["status"] = "cancelling"
            db.execute(
                "INSERT INTO jobs VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET data=excluded.data, result=COALESCE(excluded.result, jobs.result)",
                (
                    job["job_id"],
                    json.dumps(job),
                    result.model_dump_json() if result else None,
                ),
            )

    def get_job(self, job_id):
        with self.db() as db:
            row = db.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError("Unknown job_id")
        job = json.loads(row[0])
        if job.get("started_at") and job["status"] not in FINISHED:
            job["elapsed_seconds"] = round(time.time() - job["started_at"], 1)
        return job

    def list_jobs(self):
        with self.db() as db:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT data FROM jobs ORDER BY rowid DESC")
            ]

    def get_result(self, job_id):
        with self.db() as db:
            row = db.execute("SELECT result FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError("Unknown job_id")
        if row[0] is None:
            raise RuntimeError("Job has no completed result")
        return TranscriptionResult.model_validate_json(row[0])

    def submit(self, request: Submission):
        if self.stopping.is_set():
            raise RuntimeError("Service is shutting down")
        path = Path(request.audio_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError("音声ファイルが見つかりません．")
        if request.device not in DEVICE_CHOICES:
            raise ValueError("Unknown device")
        if (
            request.min_speakers
            and request.max_speakers
            and request.min_speakers > request.max_speakers
        ):
            raise ValueError("min_speakers must not exceed max_speakers")
        target_id = request.target or self.settings()["target"]
        target = (
            {"type": "local", "name": "このMac"}
            if target_id == "local"
            else {**self.profile(target_id), "type": "remote"}
        )
        job = dict(
            job_id=uuid.uuid4().hex,
            status="queued",
            target=target,
            options=request.model_dump(mode="json"),
            audio_path=str(path),
            whisper_model=request.whisper_model.value,
            diarizer_model=request.diarizer_model.value if request.diarize else None,
            diarize=request.diarize,
            engine=None,
            device=None,
            error=None,
            segments_done=0,
            segments_total=None,
            elapsed_seconds=0,
            created_at=time.time(),
            started_at=None,
        )
        self._write_job(job)
        self.executor.submit(self._run, job["job_id"], False)
        return job

    def _cancel_requested(self, job_id):
        return self.get_job(job_id).get("cancel_requested", False)

    def _finish_cancelled(self, job):
        job.update(
            status="cancelled",
            error=None,
            finished_at=time.time(),
            elapsed_seconds=round(time.time() - job["started_at"], 1)
            if job.get("started_at")
            else 0,
        )
        self._write_job(job)

    def cancel(self, job_id):
        with self.job_lock:
            job = self.get_job(job_id)
            if job["status"] in FINISHED:
                return job
            job.update(cancel_requested=True, status="cancelling")
            if not job.get("started_at"):
                self._finish_cancelled(job)
            else:
                self._write_job(job)
        if job["target"]["type"] == "local":
            with self.process_lock:
                if self.process_job_id == job_id and self.process:
                    _stop_process(self.process)
        return self.get_job(job_id)

    def _run(self, job_id, resume):
        job = self.get_job(job_id)
        backend = None
        try:
            if self.stopping.is_set() or job["status"] in FINISHED:
                return
            job.update(
                started_at=job.get("started_at") or time.time(),
                status="connecting"
                if job["target"]["type"] == "remote"
                else "loading_models",
            )
            self._write_job(job)
            if job["status"] in FINISHED:
                return
            if self._cancel_requested(job_id) and not job.get("remote_id"):
                self._finish_cancelled(job)
                return
            options = {
                k: v
                for k, v in job["options"].items()
                if k not in {"target", "audio_path", "device"}
            }
            options["whisper_model"] = WhisperModelType(options["whisper_model"])
            options["diarizer_model"] = DiarizerModelType(options["diarizer_model"])
            if job["target"]["type"] == "remote":
                backend = self.remote(job["target"])
                if not resume:
                    job["status"] = "uploading"
                    job["remote_id"] = job_id
                    self._write_job(job)
                    if self._cancel_requested(job_id):
                        self._finish_cancelled(job)
                        return
                    snapshot = backend.submit(
                        job["audio_path"], job_id=job_id, **options
                    )
                    job["remote_id"] = snapshot["job_id"]
                    self._write_job(job)
                while not self.stopping.is_set():
                    try:
                        if self._cancel_requested(job_id):
                            snapshot = backend.wait(job["remote_id"], 0)
                            if snapshot["status"] not in FINISHED:
                                snapshot = backend.cancel(job["remote_id"])
                        else:
                            snapshot = backend.wait(job["remote_id"], 2)
                        job.update(
                            {
                                k: v
                                for k, v in snapshot.items()
                                if k
                                not in {
                                    "job_id",
                                    "audio_path",
                                    "created_at",
                                    "started_at",
                                    "finished_at",
                                }
                            }
                        )
                        job["error"] = snapshot.get("error")
                        if snapshot["status"] == "done":
                            result = backend.get_result(job["remote_id"])
                            result = result.model_copy(
                                update={
                                    "job_id": job_id,
                                    "execution_target": job["target"]["url"],
                                }
                            )
                            self._write_job(job, result)
                            return
                        self._write_job(job)
                        if snapshot["status"] in FINISHED:
                            return
                        if self._cancel_requested(job_id):
                            self.stopping.wait(0.2)
                    except KeyError:
                        if self._cancel_requested(job_id):
                            self._finish_cancelled(job)
                            return
                        raise
                    except (httpx.TransportError, RuntimeError) as error:
                        # Keep the remote binding and remote_id on transient failures.
                        job.update(status="connection_lost", error=str(error))
                        self._write_job(job)
                        self.stopping.wait(2)
            else:
                self._run_local(job)
        except Exception as error:
            if job["target"]["type"] == "local" and self._cancel_requested(job_id):
                self._finish_cancelled(job)
            else:
                job.update(
                    status="interrupted" if self.stopping.is_set() else "failed",
                    error=f"{type(error).__name__}: {error}",
                )
                self._write_job(job)
        finally:
            if backend:
                backend.client.close()

    def _run_local(self, job):
        try:
            hf_token = self.credentials.get("huggingface")
        except RuntimeError:
            hf_token = None
        # Explicit environment tokens remain supported for CLI development.
        hf_token = hf_token or self.config.hf_token
        with self.process_lock:
            if self.stopping.is_set():
                return
            if self._cancel_requested(job["job_id"]):
                self._finish_cancelled(job)
                return
            process = subprocess.Popen(
                [sys.executable, "-m", "transcription_worker.local_process"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                start_new_session=True,
            )
            self.process = process
            self.process_job_id = job["job_id"]
        payload = dict(
            job_id=job["job_id"],
            options=job["options"],
            hf_token=hf_token,
            output_dir=str(self.directory / "local"),
        )
        assert process.stdin and process.stdout
        result = None
        try:
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.close()
            for line in process.stdout:
                message = json.loads(line)
                if "result" in message:
                    result = TranscriptionResult.model_validate(message["result"])
                else:
                    if self._cancel_requested(job["job_id"]):
                        continue
                    job.update(
                        {
                            k: v
                            for k, v in message.items()
                            if k not in {"job_id", "audio_path"}
                            and not (k == "status" and v in FINISHED)
                        }
                    )
                    self._write_job(job)
            code = process.wait()
            with self.job_lock:
                if self._cancel_requested(job["job_id"]):
                    self._finish_cancelled(job)
                elif code or result is None:
                    raise RuntimeError(f"Local inference process exited ({code})")
                else:
                    job.update(
                        status="done",
                        engine=result.engine,
                        device=result.device,
                        elapsed_seconds=result.elapsed_seconds,
                        error=None,
                    )
                    self._write_job(job, result)
        finally:
            with self.process_lock:
                _stop_process(process)
                self.process = None
                self.process_job_id = None
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()

    def request_stop(self):
        self.stopping.set()
        with self.process_lock:
            process = self.process
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

    def close(self):
        self.request_stop()
        self.executor.shutdown(wait=True, cancel_futures=True)
        for job in self.list_jobs():
            if job["status"] not in FINISHED and job["target"]["type"] == "local":
                job.update(
                    status="interrupted", error="サービス終了により中断しました．"
                )
                self._write_job(job)


def create_app(service: FrontendService, token: str, shutdown=lambda: None):
    app = FastAPI(title="noteit desktop service")
    bearer = HTTPBearer(auto_error=False)

    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credentials is None or not secrets.compare_digest(
            credentials.credentials, token
        ):
            raise HTTPException(401, "Invalid desktop token")

    app.router.dependencies.append(Depends(authorize))

    @app.exception_handler(KeyError)
    async def missing(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.exception_handler(ValueError)
    @app.exception_handler(RuntimeError)
    @app.exception_handler(httpx.HTTPError)
    async def invalid(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.get("/health")
    def health():
        if service.stopping.is_set():
            raise HTTPException(503, "Service is stopping")
        return {"status": "ok", "service": "noteit-frontend", "version": 1}

    @app.get("/settings")
    def settings():
        return service.settings()

    @app.put("/settings")
    def save_settings(body: Settings):
        return service.save_settings(body)

    @app.get("/profiles")
    def profiles():
        return service.profiles()

    @app.post("/profiles")
    def save_profile(body: Profile):
        return service.save_profile(body)

    @app.delete("/profiles/{profile_id}")
    def delete_profile(profile_id: str):
        service.delete_profile(profile_id)
        return {"ok": True}

    @app.post("/profiles/{profile_id}/test")
    def test_profile(profile_id: str):
        return service.describe(profile_id)

    @app.get("/models")
    def models(target: str | None = None):
        return service.describe(target)

    @app.post("/jobs")
    def submit(body: Submission):
        return service.submit(body)

    @app.get("/jobs")
    def jobs():
        return [service.get_job(job["job_id"]) for job in service.list_jobs()]

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        return service.get_job(job_id)

    @app.get("/jobs/{job_id}/result")
    def result(job_id: str):
        return service.get_result(job_id)

    @app.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return service.cancel(job_id)

    @app.get("/jobs/{job_id}/text")
    def text(job_id: str):
        return {"text": format_transcript(service.get_result(job_id).utterances)}

    @app.post("/shutdown")
    def stop():
        service.request_stop()
        shutdown()
        return {"ok": True}

    return app


def read_endpoint(directory):
    try:
        endpoint = json.loads((directory / "endpoint.json").read_text())
        if urlsplit(endpoint["url"]).hostname != "127.0.0.1":
            return None
        response = httpx.get(
            endpoint["url"] + "/health",
            headers={"Authorization": "Bearer " + endpoint["token"]},
            timeout=1,
        )
        if (
            response.status_code == 200
            and response.json().get("service") == "noteit-frontend"
        ):
            return endpoint
    except (OSError, ValueError, KeyError, httpx.HTTPError):
        pass
    return None


def ensure_service():
    directory = data_directory()
    with (directory / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        endpoint = read_endpoint(directory)
        if endpoint:
            return endpoint
        log_path = directory / "service.log"
        log_path.touch(mode=0o600)
        log_path.chmod(0o600)
        with log_path.open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "transcription_worker.frontend", "run"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        for _ in range(150):
            endpoint = read_endpoint(directory)
            if endpoint:
                return endpoint
            if process.poll() is not None:
                raise RuntimeError(f"Desktop service failed to start. See {log_path}")
            time.sleep(0.1)
        raise RuntimeError(f"Desktop service startup timed out. See {log_path}")


def run_service():
    import uvicorn

    directory = data_directory()
    with (directory / "instance.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        token = secrets.token_urlsafe(32)
        service = FrontendService(directory)
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(
                    service, token, lambda: setattr(server, "should_exit", True)
                ),
                log_level="warning",
            )
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            endpoint = {
                "url": f"http://127.0.0.1:{sock.getsockname()[1]}",
                "token": token,
            }
            path = directory / "endpoint.json"
            temporary = path.with_suffix(".tmp")
            temporary.touch(mode=0o600)
            temporary.write_text(json.dumps(endpoint))
            temporary.replace(path)
            try:
                server.run(sockets=[sock])
            finally:
                service.close()
                path.unlink(missing_ok=True)


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["ensure", "run"], default="ensure", nargs="?"
    )
    args = parser.parse_args()
    if args.command == "ensure":
        print(json.dumps(ensure_service()))
    else:
        run_service()


if __name__ == "__main__":
    main()
