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
    def from_env(cls) -> ServiceConfig:
        whisper_model = os.environ.get("NOTEIT_WHISPER_MODEL")
        return cls(
            hf_token=os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
            or os.environ.get("HF_TOKEN"),
            whisper_model=WhisperModelType(whisper_model)
            if whisper_model
            else default_whisper_model(),
            diarizer_model=DiarizerModelType(
                os.environ.get("NOTEIT_DIARIZER_MODEL", DiarizerModelType.COMMUNITY)
            ),
            device=os.environ.get("NOTEIT_DEVICE", "auto"),
            diarizer_device=os.environ.get("NOTEIT_DIARIZER_DEVICE", "auto"),
            compile_whisper=os.environ.get("WHISPER_COMPILE", "1").lower()
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
    LOADING_MODELS = "loading_models"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    DONE = "done"
    FAILED = "failed"


class TranscriptionResult(BaseModel):
    job_id: str
    audio_path: str
    language: str | None
    whisper_model: WhisperModelType
    diarizer_model: DiarizerModelType
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
    diarizer_model: DiarizerModelType
    speaker_hints: dict[str, int]
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
            "audio_path": str(self.audio_path),
            "whisper_model": self.whisper_model.value,
            "diarizer_model": self.diarizer_model.value,
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

    # ---- public API -----------------------------------------------------

    def submit(
        self,
        audio_path: str | Path,
        *,
        language: str | None = "ja",
        whisper_model: WhisperModelType | None = None,
        diarizer_model: DiarizerModelType | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> Job:
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
            id=uuid.uuid4().hex[:12],
            audio_path=path,
            language=language or None,
            whisper_model=whisper_model or self.config.whisper_model,
            diarizer_model=diarizer_model or self.config.diarizer_model,
            speaker_hints=hints,
        )
        self._jobs[job.id] = job
        self._executor.submit(self._run, job)
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

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
            if job.status is JobStatus.FAILED:
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
        job = self._jobs.get(job_id)
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
            job.status = JobStatus.LOADING_MODELS
            worker = self._load_worker(job.whisper_model, job.diarizer_model)

            def on_diarized(segments: list[Any]) -> None:
                job.segments_total = len(segments)
                job.status = JobStatus.TRANSCRIBING

            def on_segment(_segment: Any) -> None:
                job.segments_done += 1

            job.status = JobStatus.DIARIZING
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
                audio_path=str(job.audio_path),
                language=job.language,
                whisper_model=job.whisper_model,
                diarizer_model=job.diarizer_model,
                elapsed_seconds=round(time.time() - job.started_at, 1),
                speakers=sorted({u.speaker_label for u in utterances}),
                utterances=utterances,
                words=words,
            )
            self._save(job.result)
            job.status = JobStatus.DONE
        except Exception as error:  # noqa: BLE001 - reported back to the caller
            job.error = f"{type(error).__name__}: {error}"
            job.status = JobStatus.FAILED
        finally:
            job.finished_at = time.time()
            job.finished.set()

    def _save(self, result: TranscriptionResult) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self._result_path(result.job_id)
        path.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    def _load_worker(
        self, whisper_model: WhisperModelType, diarizer_model: DiarizerModelType
    ) -> TranscriptionWorker:
        # Imported lazily so the MCP server starts instantly and tests do not
        # need torch/pyannote model weights.
        from .speaker_diarizer import SpeakerDiarizer
        from .whisper import Whisper
        from .worker import TranscriptionWorker

        if not self.config.hf_token:
            raise RuntimeError(
                "HUGGINGFACE_ACCESS_TOKEN is not set. pyannote models require it."
            )

        if self._diarizer is None or self._diarizer.model_type != diarizer_model:
            self._diarizer = None
            _free_memory()
            self._diarizer = SpeakerDiarizer(
                hf_token=self.config.hf_token,
                model_type=diarizer_model,
                device=self.config.diarizer_device,
            )
        if self._whisper is None or self._whisper.model_type != whisper_model:
            self._whisper = None
            _free_memory()
            self._whisper = Whisper(
                hf_token=self.config.hf_token,
                model_type=whisper_model,
                device=self.config.device,
                compile_model=self.config.compile_whisper,
            )
        return TranscriptionWorker(diarizer=self._diarizer, whisper=self._whisper)


def _free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass
