"""HTTP job queue whose inference processes never survive their jobs."""

from dataclasses import asdict
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time

from .service import Job, JobStatus, TranscriptionResult, TranscriptionService

PROCESS_STOP_TIMEOUT = 5
FINISHED = {
    JobStatus.DONE,
    JobStatus.FAILED,
    JobStatus.INTERRUPTED,
    JobStatus.CANCELLED,
}


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=PROCESS_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


class ProcessTranscriptionService(TranscriptionService):
    """Keep the API and durable queue alive, but cold-start every inference job."""

    def __init__(self, config):
        super().__init__(config)
        self._stopping = threading.Event()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._process_job_id: str | None = None

    def cancel(self, job_id: str) -> Job:
        with self._process_lock:
            job = self._require_job(job_id)
            if job.status in FINISHED:
                return job
            job.cancel_requested.set()
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                if job.delete_audio_after:
                    job.audio_path.unlink(missing_ok=True)
                job.finished_at = time.time()
                self._persist(job)
                job.finished.set()
            else:
                job.status = JobStatus.CANCELLING
                self._persist(job)
                if self._process_job_id == job_id and self._process:
                    _stop_process(self._process)
            return job

    def _submit(self, audio_path, **options):
        if self._stopping.is_set():
            raise RuntimeError("Inference service is shutting down")
        return super()._submit(audio_path, **options)

    def _run(self, job: Job) -> None:
        with self._process_lock:
            if job.finished.is_set():
                return
            job.started_at = time.time()
            job.status = JobStatus.LOADING_MODELS
            self._persist(job)
        try:
            if self._stopping.is_set():
                raise RuntimeError("Interrupted by service shutdown")
            if job.cancel_requested.is_set():
                raise RuntimeError("Cancelled")
            temporary = Path(
                os.environ.get(
                    "NOTEIT_TEMP_DIR", self.config.output_dir.parent / "temporary"
                )
            )
            temporary.mkdir(parents=True, exist_ok=True, mode=0o700)
            with TemporaryDirectory(
                prefix="noteit-transcription-", dir=temporary
            ) as name:
                directory = Path(name)
                config = {
                    **asdict(self.config),
                    "output_dir": str(directory / "results"),
                }
                payload = {
                    "config": config,
                    "audio_path": str(job.audio_path),
                    "options": {
                        "job_id": job.id,
                        "language": job.language,
                        "whisper_model": job.whisper_model,
                        "diarizer_model": job.diarizer_model,
                        "diarize": job.diarizer_model is not None,
                        "source_name": job.source_name,
                        **job.speaker_hints,
                    },
                }
                decoded = directory / "temporary"
                decoded.mkdir(mode=0o700)
                result = self._infer(
                    job, payload, {**os.environ, "NOTEIT_TEMP_DIR": str(decoded)}
                )
            # _infer has reaped the process: no model/CUDA context remains alive.
            with self._process_lock:
                if job.cancel_requested.is_set():
                    raise RuntimeError("Cancelled")
                job.result = result.model_copy(
                    update={"elapsed_seconds": round(time.time() - job.started_at, 1)}
                )
                self._save(job.result)
                job.status = JobStatus.DONE
        except Exception as error:
            job.status = (
                JobStatus.CANCELLED
                if job.cancel_requested.is_set()
                else JobStatus.INTERRUPTED
                if self._stopping.is_set()
                else JobStatus.FAILED
            )
            job.error = (
                None
                if job.status == JobStatus.CANCELLED
                else f"{type(error).__name__}: {error}"
            )
        finally:
            if job.delete_audio_after:
                job.audio_path.unlink(missing_ok=True)
            job.finished_at = time.time()
            self._persist(job)
            job.finished.set()

    def _infer(self, job, payload, environment) -> TranscriptionResult:
        with self._process_lock:
            if self._stopping.is_set():
                raise RuntimeError("Interrupted by service shutdown")
            if job.cancel_requested.is_set():
                raise RuntimeError("Cancelled")
            process = subprocess.Popen(
                [sys.executable, "-m", "transcription_worker.inference_process"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                start_new_session=True,
                env=environment,
            )
            self._process = process
            self._process_job_id = job.id
        result = None
        error = None
        try:
            assert process.stdin and process.stdout
            # Credentials travel over stdin, never argv or an on-disk job file.
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.close()
            for line in process.stdout:
                message = json.loads(line)
                if "result" in message:
                    result = TranscriptionResult.model_validate(message["result"])
                    continue
                status = JobStatus(message["status"])
                with self._process_lock:
                    if not job.cancel_requested.is_set():
                        if status not in FINISHED:
                            job.status = status
                        error = message.get("error") or error
                        for key in (
                            "engine",
                            "device",
                            "segments_done",
                            "segments_total",
                        ):
                            if key in message:
                                setattr(job, key, message[key])
                        self._persist(job)
            code = process.wait()
            if job.cancel_requested.is_set():
                raise RuntimeError("Cancelled")
            if self._stopping.is_set():
                raise RuntimeError("Interrupted by service shutdown")
            if code:
                raise RuntimeError(f"Inference process exited ({code})")
            if error:
                raise RuntimeError(error)
            if result is None or result.job_id != job.id:
                raise RuntimeError("Inference process exited without a valid result")
            return result
        finally:
            with self._process_lock:
                _stop_process(process)
                self._process = None
                self._process_job_id = None
            for stream in (process.stdin, process.stdout):
                if stream:
                    with contextlib.suppress(OSError):
                        stream.close()

    def close(self):
        with self._submit_lock:
            self._stopping.set()
        with self._process_lock:
            if self._process:
                _stop_process(self._process)
        # Drain queued jobs as interrupted, including removal of owned uploads.
        super().close()
