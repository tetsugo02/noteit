"""Where transcription runs: in this process, or on a remote inference server.

The MCP server and CLI talk to a ``TranscriptionBackend`` and do not care which
one they get. Set ``NOTEIT_SERVER_URL`` (and ``NOTEIT_API_TOKEN``) to use a
remote GPU server; otherwise models run locally.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import httpx

from .models import (
    DIARIZER_MODEL_INFO,
    WHISPER_MODEL_INFO,
    DiarizerModelType,
    WhisperModelType,
    engine_capabilities,
)
from .service import (
    JobStatus,
    ServiceConfig,
    TranscriptionResult,
    TranscriptionService,
)

FINISHED = {JobStatus.DONE.value, JobStatus.FAILED.value, JobStatus.INTERRUPTED.value}


class TranscriptionBackend(Protocol):
    def describe(self) -> dict[str, Any]: ...

    def submit(
        self,
        audio_path: str | Path,
        *,
        language: str | None = "ja",
        whisper_model: WhisperModelType | None = None,
        diarizer_model: DiarizerModelType | None = None,
        diarize: bool = True,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> dict[str, Any]: ...

    def wait(self, job_id: str, timeout: float) -> dict[str, Any]: ...

    def get_result(self, job_id: str) -> TranscriptionResult: ...

    def list_jobs(self) -> list[dict[str, Any]]: ...


def model_catalog(
    whisper_default: WhisperModelType, diarizer_default: DiarizerModelType
) -> dict[str, Any]:
    return {
        **engine_capabilities(),
        "whisper_models": [
            {"name": m.value, "parameters": i.parameters, "note": i.note}
            for m, i in WHISPER_MODEL_INFO.items()
        ],
        "diarizer_models": [
            {"name": m.value, "note": i.note} for m, i in DIARIZER_MODEL_INFO.items()
        ],
        "defaults": {
            "whisper_model": whisper_default.value,
            "diarizer_model": diarizer_default.value,
        },
    }


class LocalBackend:
    def __init__(self, service: TranscriptionService) -> None:
        self.service = service

    def describe(self) -> dict[str, Any]:
        config = self.service.config
        catalog = model_catalog(config.whisper_model, config.diarizer_model)
        catalog["backend"] = {"type": "local", "device": config.device}
        return catalog

    def submit(self, audio_path: str | Path, **options: Any) -> dict[str, Any]:
        return self.service.submit(audio_path, **options).snapshot()

    def wait(self, job_id: str, timeout: float) -> dict[str, Any]:
        return self.service.wait(job_id, timeout).snapshot()

    def get_result(self, job_id: str) -> TranscriptionResult:
        return self.service.get_result(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs = [job.snapshot() for job in self.service.list_jobs()]
        known = {job["job_id"] for job in jobs}
        jobs += [
            {"job_id": path.stem, "status": JobStatus.DONE.value}
            for path in self.service.list_saved_results()
            if path.stem not in known
        ]
        return jobs


def compress_for_upload(source: Path, destination: Path) -> None:
    """Re-encode to 16 kHz mono Opus: what the models use, ~10x smaller."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to upload audio to the server.")
    subprocess.run(
        [ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "32k",
         str(destination)],
        check=True,
    )  # fmt: skip


def job_key(audio_path: Path, options: dict[str, Any]) -> str:
    """Hash content, options and pipeline revision for idempotent uploads."""
    digest = hashlib.sha256()
    # Do not reuse transcripts made by the former per-turn Transformers pipeline.
    digest.update(b"noteit:faster-whisper-full-recording:v1\0")
    with audio_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    digest.update(json.dumps(options, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


class RemoteBackend:
    def __init__(
        self,
        base_url: str,
        token: str | None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        # Per-request timeouts are skipped for injected clients (e.g. TestClient).
        self.send_timeouts = client is None
        self.client = client or httpx.Client(base_url=base_url)
        self.client.headers.update(headers)
        self.base_url = base_url

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if not self.send_timeouts:
            kwargs.pop("timeout", None)
        response = self.client.request(method, path, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            if response.status_code == 404:
                raise KeyError(detail)
            raise RuntimeError(f"Server error {response.status_code}: {detail}")
        return response.json()

    def describe(self) -> dict[str, Any]:
        catalog = self._request("GET", "/models")
        catalog["backend"] = {
            "type": "remote",
            "url": self.base_url,
            **catalog.get("backend", {}),
        }
        return catalog

    def submit(
        self,
        audio_path: str | Path,
        *,
        language: str | None = "ja",
        whisper_model: WhisperModelType | None = None,
        diarizer_model: DiarizerModelType | None = None,
        diarize: bool = True,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> dict[str, Any]:
        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        options = {
            key: (value.value if hasattr(value, "value") else value)
            for key, value in {
                "language": language,
                "whisper_model": whisper_model,
                "diarizer_model": diarizer_model,
                "diarize": diarize,
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }.items()
            if value is not None
        }
        job_id = job_key(path, options)

        # Skip the upload when the server already has (or is running) this job.
        try:
            existing = self._request("GET", f"/jobs/{job_id}")
            if existing["status"] not in {
                JobStatus.FAILED.value,
                JobStatus.INTERRUPTED.value,
            }:
                return existing
        except KeyError:
            pass

        with TemporaryDirectory(prefix="noteit-upload-") as temporary_directory:
            compressed = Path(temporary_directory) / "audio.ogg"
            compress_for_upload(path, compressed)
            with compressed.open("rb") as file:
                return self._request(
                    "POST",
                    "/jobs",
                    data={**options, "job_id": job_id, "source_name": path.name},
                    files={"audio": ("audio.ogg", file, "audio/ogg")},
                    timeout=httpx.Timeout(30.0, write=600.0),
                )

    def wait(self, job_id: str, timeout: float) -> dict[str, Any]:
        wait = max(0, min(60, int(timeout)))
        return self._request(
            "GET", f"/jobs/{job_id}", params={"wait": wait}, timeout=wait + 30.0
        )

    def get_result(self, job_id: str) -> TranscriptionResult:
        return TranscriptionResult.model_validate(
            self._request("GET", f"/jobs/{job_id}/result", timeout=120.0)
        )

    def list_jobs(self) -> list[dict[str, Any]]:
        return self._request("GET", "/jobs")


def backend_from_env() -> TranscriptionBackend:
    url = os.environ.get("NOTEIT_SERVER_URL")
    if url:
        return RemoteBackend(url, os.environ.get("NOTEIT_API_TOKEN"))
    return LocalBackend(TranscriptionService(ServiceConfig.from_env()))
