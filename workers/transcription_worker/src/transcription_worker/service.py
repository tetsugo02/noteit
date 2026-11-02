"""Job queue and model cache shared by the MCP server and the CLI.

Models are loaded lazily and kept in memory between jobs. Requesting a
different model size swaps only the component that changed. Jobs run one at a
time on a background thread, and finished results are written to disk so they
survive a server restart.
"""

from __future__ import annotations

import gc
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .models import DiarizerModelType, WhisperModelType, default_whisper_model
from .schema import SpeakerAssignedWord
from .transcript import Utterance, group_utterances

if TYPE_CHECKING:
    from .speaker_diarizer import SpeakerDiarizer
    from .whisper import Whisper
    from .worker import TranscriptionWorker


@dataclass(frozen=True)
class ServiceConfig:
    hf_token: str | None
    whisper_model: WhisperModelType
    diarizer_model: DiarizerModelType
    device: str
    diarizer_device: str
    compile_whisper: bool
    output_dir: Path

    @classmethod
    def from_env(
        cls, *, default_model: WhisperModelType | None = None
    ) -> ServiceConfig:
        whisper_model = os.environ.get("NOTEIT_WHISPER_MODEL")
        return cls(
            hf_token=os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
            or os.environ.get("HF_TOKEN"),
            whisper_model=WhisperModelType(whisper_model)
            if whisper_model
            else (default_model or default_whisper_model()),
            diarizer_model=DiarizerModelType(
                os.environ.get("NOTEIT_DIARIZER_MODEL", DiarizerModelType.COMMUNITY)
            ),
            device=os.environ.get("NOTEIT_DEVICE", "auto"),
            diarizer_device=os.environ.get("NOTEIT_DIARIZER_DEVICE", "auto"),
            compile_whisper=os.environ.get("WHISPER_COMPILE", "0").lower()
            not in {"0", "false", "no"},
            output_dir=Path(
                os.environ.get(
                    "NOTEIT_OUTPUT_DIR",
                    Path.home() / ".cache" / "noteit" / "transcripts",
                )
            ).expanduser(),
        )


class JobStatus(StrEnum):
    QUEUED = "queued"
    DOWNLOADING_MODEL = "downloading_model"
    LOADING_MODELS = "loading_models"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    DONE = "done"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class TranscriptionResult(BaseModel):
    job_id: str
    audio_path: str
    language: str | None
    whisper_model: WhisperModelType
    diarizer_model: DiarizerModelType | None
    engine: str | None = None
    device: str | None = None
    execution_target: str | None = None
    elapsed_seconds: float
    speakers: list[str]
    utterances: list[Utterance]
    words: list[SpeakerAssignedWord]


@dataclass
class Job:
    id: str
    audio_path: Path
    language: str | None
    whisper_model: WhisperModelType
    diarizer_model: DiarizerModelType | None
    speaker_hints: dict[str, int]
    engine: str | None = None
    device: str | None = None
    delete_audio_after: bool = False
    source_name: str | None = None
    status: JobStatus = JobStatus.QUEUED
    segments_total: int | None = None
    segments_done: int = 0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: TranscriptionResult | None = None
    finished: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, Any]:
        now = self.finished_at or time.time()
        return {
            "job_id": self.id,
            "status": self.status.value,
            "audio_path": self.source_name or str(self.audio_path),
            "whisper_model": self.whisper_model.value,
            "diarizer_model": self.diarizer_model.value
            if self.diarizer_model
            else None,
            "diarize": self.diarizer_model is not None,
            "engine": self.engine,
            "device": self.device,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "segments_done": self.segments_done,
            "segments_total": self.segments_total,
            "elapsed_seconds": round(now - self.started_at, 1)
            if self.started_at
            else 0.0,
            "error": self.error,
        }


WorkerFactory = Callable[["SpeakerDiarizer", "Whisper"], "TranscriptionWorker"]


class TranscriptionService:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self._jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="noteit")
        self._diarizer: SpeakerDiarizer | None = None
        self._whisper: Whisper | None = None
        self._submit_lock = threading.Lock()
        self._state_callback = lambda state: None
        config.output_dir.mkdir(parents=True, exist_ok=True)
        with self._database() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
            )
            rows = db.execute("SELECT data FROM jobs").fetchall()
        for (data,) in rows:
            item = json.loads(data)
            status = JobStatus(item["status"])
            if status not in {JobStatus.DONE, JobStatus.FAILED, JobStatus.INTERRUPTED}:
                status = JobStatus.INTERRUPTED
            job = Job(
                id=item["job_id"],
                audio_path=Path(item["audio_path"]),
                language=None,
                whisper_model=WhisperModelType(item["whisper_model"]),
                diarizer_model=DiarizerModelType(item["diarizer_model"])
                if item["diarizer_model"]
                else None,
                speaker_hints={},
                status=status,
                engine=item.get("engine"),
                device=item.get("device"),
                error="Interrupted by service restart"
                if status == JobStatus.INTERRUPTED
                else item.get("error"),
                created_at=item["created_at"],
                started_at=item.get("started_at"),
                finished_at=item.get("finished_at") or time.time(),
            )
            if status == JobStatus.DONE and self._result_path(job.id).exists():
                job.result = TranscriptionResult.model_validate_json(
                    self._result_path(job.id).read_text()
                )
            job.finished.set()
            self._jobs[job.id] = job
            self._persist(job)

    def _database(self):
        return sqlite3.connect(self.config.output_dir / "jobs.sqlite3", timeout=30)

    def _persist(self, job):
        with self._database() as db:
            db.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?, ?)",
                (job.id, json.dumps(job.snapshot())),
            )

    def submit(self, audio_path: str | Path, **options) -> Job:
        with self._submit_lock:
            return self._submit(audio_path, **options)

    # ---- public API -----------------------------------------------------

    def _submit(
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
        job_id: str | None = None,
        source_name: str | None = None,
        delete_audio_after: bool = False,
    ) -> Job:
        """Queue a job. A caller-chosen ``job_id`` makes resubmission idempotent:
        an existing job or saved result with that id is reused."""
        if job_id is not None:
            if not job_id.isalnum():
                raise ValueError(f"Invalid job_id: {job_id}")
            existing = self._jobs.get(job_id)
            if existing is not None and existing.status not in {
                JobStatus.FAILED,
                JobStatus.INTERRUPTED,
            }:
                return existing
            if self._result_path(job_id).is_file():
                return self._job_from_saved(job_id)

        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        hints = {
            key: value
            for key, value in {
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }.items()
            if value is not None
        }
        job = Job(
            id=job_id or uuid.uuid4().hex[:12],
            audio_path=path,
            language=language or None,
            whisper_model=whisper_model or self.config.whisper_model,
            diarizer_model=(diarizer_model or self.config.diarizer_model)
            if diarize
            else None,
            speaker_hints=hints,
            delete_audio_after=delete_audio_after,
            source_name=source_name,
        )
        self._jobs[job.id] = job
        self._persist(job)
        self._executor.submit(self._run, job)
        return job

    def get_job(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None and job_id.isalnum() and self._result_path(job_id).is_file():
            job = self._job_from_saved(job_id)
        return job

    def list_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def wait(self, job_id: str, timeout: float | None) -> Job:
        job = self._require_job(job_id)
        job.finished.wait(timeout)
        return job

    def get_result(self, job_id: str) -> TranscriptionResult:
        """Return a finished result from memory, or from disk after a restart."""
        job = self._jobs.get(job_id)
        if job is not None:
            if job.result is not None:
                return job.result
            if job.status in {JobStatus.FAILED, JobStatus.INTERRUPTED}:
                raise RuntimeError(f"Job {job_id} failed: {job.error}")
            raise RuntimeError(
                f"Job {job_id} is not finished yet (status: {job.status})."
            )

        path = self._result_path(job_id)
        if not path.is_file():
            raise KeyError(f"Unknown job_id: {job_id}")
        return TranscriptionResult.model_validate_json(path.read_text(encoding="utf-8"))

    def list_saved_results(self) -> list[Path]:
        if not self.config.output_dir.is_dir():
            return []
        return sorted(
            self.config.output_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    # ---- internals ------------------------------------------------------

    def _require_job(self, job_id: str) -> Job:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        return job

    def _result_path(self, job_id: str) -> Path:
        # job ids are generated hex strings; reject anything else to keep
        # lookups inside the output directory.
        if not job_id.isalnum():
            raise KeyError(f"Unknown job_id: {job_id}")
        return self.config.output_dir / f"{job_id}.json"

    def _run(self, job: Job) -> None:
        job.started_at = time.time()
        try:

            def state(value):
                job.status = JobStatus(value)
                self._persist(job)

            self._state_callback = state
            state(JobStatus.LOADING_MODELS)
            worker = self._load_worker(job.whisper_model, job.diarizer_model)

            job.engine = (
                getattr(worker._whisper, "engine", None)
                if hasattr(worker, "_whisper")
                else None
            )
            job.device = (
                getattr(worker._whisper, "device", self.config.device)
                if hasattr(worker, "_whisper")
                else self.config.device
            )
            if job.engine is None:
                job.engine = "mlx-whisper" if job.device == "mlx" else "faster-whisper"

            def on_diarized(segments: list[Any]) -> None:
                job.segments_total = len(segments)
                state(JobStatus.TRANSCRIBING)

            def on_segment(_segment: Any) -> None:
                job.segments_done += 1

            state(JobStatus.DIARIZING if job.diarizer_model else JobStatus.TRANSCRIBING)
            words = worker.transcribe(
                job.audio_path,
                language=job.language,
                on_diarized=on_diarized,
                on_segment=on_segment,
                **job.speaker_hints,
            )
            utterances = group_utterances(words)
            job.result = TranscriptionResult(
                job_id=job.id,
                audio_path=job.source_name or str(job.audio_path),
                language=job.language,
                whisper_model=job.whisper_model,
                diarizer_model=job.diarizer_model,
                elapsed_seconds=round(time.time() - job.started_at, 1),
                speakers=sorted(
                    {u.speaker_label for u in utterances if u.speaker_label}
                ),
                engine=job.engine,
                device=job.device,
                execution_target="local",
                utterances=utterances,
                words=words,
            )
            self._save(job.result)
            job.status = JobStatus.DONE
        except Exception as error:  # noqa: BLE001 - reported back to the caller
            job.error = f"{type(error).__name__}: {error}"
            job.status = JobStatus.FAILED
        finally:
            if job.delete_audio_after:
                job.audio_path.unlink(missing_ok=True)
            job.finished_at = time.time()
            self._persist(job)
            job.finished.set()

    def _job_from_saved(self, job_id: str) -> Job:
        """Rebuild a finished Job from its saved result (e.g. after a restart)."""
        result = TranscriptionResult.model_validate_json(
            self._result_path(job_id).read_text()
        )
        job = Job(
            id=job_id,
            audio_path=Path(result.audio_path),
            language=result.language,
            whisper_model=result.whisper_model,
            diarizer_model=result.diarizer_model,
            speaker_hints={},
            status=JobStatus.DONE,
            engine=result.engine,
            device=result.device,
            result=result,
        )
        job.started_at = job.created_at - result.elapsed_seconds
        job.finished_at = job.created_at
        job.finished.set()
        self._jobs[job_id] = job
        return job

    def _save(self, result: TranscriptionResult) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self._result_path(result.job_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

        temporary.replace(path)

    def _load_worker(
        self, whisper_model: WhisperModelType, diarizer_model: DiarizerModelType | None
    ) -> TranscriptionWorker:
        # Imported lazily so the MCP server starts instantly and tests do not
        # need torch/pyannote model weights.
        from .models import recognition_device
        from .worker import TranscriptionWorker

        if diarizer_model:
            from .speaker_diarizer import SpeakerDiarizer

            if not self.config.hf_token:
                raise RuntimeError("Hugging Face token is required for diarization.")
            if self._diarizer is None or self._diarizer.model_type != diarizer_model:
                self._diarizer = None
                _free_memory()
                self._state_callback("downloading_model")
                self._diarizer = SpeakerDiarizer(
                    hf_token=self.config.hf_token,
                    model_type=diarizer_model,
                    device=self.config.diarizer_device,
                )
        engine, device = recognition_device(self.config.device)
        if self._whisper is None or self._whisper.model_type != whisper_model:
            self._whisper = None
            _free_memory()
            if engine == "mlx-whisper":
                from .mlx_engine import MLXWhisper

                self._whisper = MLXWhisper(whisper_model, self._state_callback)
            else:
                from .whisper import Whisper

                self._whisper = Whisper(
                    hf_token=self.config.hf_token,
                    model_type=whisper_model,
                    device=device,
                    compile_model=self.config.compile_whisper,
                    on_state=self._state_callback,
                )
        return TranscriptionWorker(
            diarizer=self._diarizer if diarizer_model else None, whisper=self._whisper
        )


def _free_memory() -> None:
    gc.collect()
    if "torch" not in sys.modules:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass
