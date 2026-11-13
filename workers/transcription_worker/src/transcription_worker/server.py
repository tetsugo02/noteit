"""HTTP inference server for running pyannote + Whisper on a GPU machine.

Intended setup: bind to 127.0.0.1 on the GPU server and reach it through an
SSH tunnel. A bearer token is still required because other users of a shared
server can connect to its localhost ports.

    NOTEIT_API_TOKEN=... uv run --extra server noteit-server --port 8765
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
import os
import secrets
import shutil
from pathlib import Path
import uuid
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .backends import model_catalog
from .models import DiarizerModelType, WhisperModelType
from .process_service import ProcessTranscriptionService
from .service import JobStatus, ServiceConfig, TranscriptionService

MAX_WAIT_SECONDS = 60
UPLOAD_CHUNK_BYTES = 1 << 20


def create_app(
    service: TranscriptionService, token: str, max_upload_mb: int = 1024
) -> FastAPI:
    if not token:
        raise ValueError("An API token is required.")

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await asyncio.to_thread(service.close)

    app = FastAPI(title="noteit transcription server", lifespan=lifespan)
    bearer = HTTPBearer(auto_error=False)
    upload_dir = service.config.output_dir.parent / "uploads"
    for directory in (service.config.output_dir, upload_dir):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)  # meeting audio and transcripts are private
    temporary = Path(
        os.environ.get(
            "NOTEIT_TEMP_DIR", service.config.output_dir.parent / "temporary"
        )
    )
    temporary.mkdir(parents=True, exist_ok=True, mode=0o700)
    for path in temporary.glob("noteit-transcription-*"):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)

    os.environ["NOTEIT_TEMP_DIR"] = str(temporary)
    for path in upload_dir.glob("*.upload"):
        path.unlink(missing_ok=True)

    def authorize(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        if credentials is None or not secrets.compare_digest(
            credentials.credentials.encode(), token.encode()
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing API token.")

    def find_job(job_id: str):
        job = service.get_job(job_id) if job_id.isalnum() else None
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
        return job

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models", dependencies=[Depends(authorize)])
    def models() -> dict[str, Any]:
        config = service.config
        catalog = model_catalog(config.whisper_model, config.diarizer_model)
        catalog["backend"] = {"device": config.device}
        return catalog

    @app.post("/jobs", dependencies=[Depends(authorize)])
    async def create_job(
        audio: Annotated[UploadFile, File()],
        job_id: Annotated[str | None, Form(pattern=r"^[A-Za-z0-9]{1,64}$")] = None,
        source_name: Annotated[str | None, Form(max_length=255)] = None,
        language: Annotated[str | None, Form()] = "ja",
        whisper_model: Annotated[WhisperModelType | None, Form()] = None,
        diarizer_model: Annotated[DiarizerModelType | None, Form()] = None,
        diarize: Annotated[bool, Form()] = True,
        num_speakers: Annotated[int | None, Form(ge=1)] = None,
        min_speakers: Annotated[int | None, Form(ge=1)] = None,
        max_speakers: Annotated[int | None, Form(ge=1)] = None,
    ) -> dict[str, Any]:
        if job_id is not None:
            existing = service.get_job(job_id)
            if existing is not None and existing.status not in {
                JobStatus.FAILED,
                JobStatus.INTERRUPTED,
                JobStatus.CANCELLED,
            }:
                return existing.snapshot()

        upload_path = upload_dir / f"{uuid.uuid4().hex}.upload"
        limit = max_upload_mb * 1024 * 1024
        written = 0
        try:
            with upload_path.open("wb") as file:
                while chunk := await audio.read(UPLOAD_CHUNK_BYTES):
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload exceeds {max_upload_mb} MB.",
                        )
                    file.write(chunk)
            job = service.submit(
                upload_path,
                language=language or None,
                whisper_model=whisper_model,
                diarizer_model=diarizer_model,
                diarize=diarize,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                job_id=job_id,
                source_name=source_name,
                delete_audio_after=True,
            )
        except BaseException:
            upload_path.unlink(missing_ok=True)
            raise
        if job.audio_path != upload_path.resolve():
            upload_path.unlink(missing_ok=True)
        return job.snapshot()

    @app.get("/jobs", dependencies=[Depends(authorize)])
    def list_jobs() -> list[dict[str, Any]]:
        jobs = [job.snapshot() for job in service.list_jobs()]
        known = {job["job_id"] for job in jobs}
        return jobs + [
            {"job_id": path.stem, "status": JobStatus.DONE.value}
            for path in service.list_saved_results()
            if path.stem not in known
        ]

    @app.get("/jobs/{job_id}", dependencies=[Depends(authorize)])
    async def get_job(
        job_id: str,
        wait: Annotated[int, Query(ge=0, le=MAX_WAIT_SECONDS)] = 0,
    ) -> dict[str, Any]:
        job = find_job(job_id)
        if wait:
            await asyncio.to_thread(job.finished.wait, wait)
        return job.snapshot()

    @app.get("/jobs/{job_id}/result", dependencies=[Depends(authorize)])
    def get_result(job_id: str) -> dict[str, Any]:
        job = find_job(job_id)
        if job.status in {JobStatus.FAILED, JobStatus.INTERRUPTED, JobStatus.CANCELLED}:
            raise HTTPException(status_code=409, detail=f"Job failed: {job.error}")
        if job.result is None:
            raise HTTPException(
                status_code=409,
                detail=f"Job not finished (status: {job.status.value}).",
            )
        return job.result.model_dump(mode="json")

    @app.post("/jobs/{job_id}/cancel", dependencies=[Depends(authorize)])
    def cancel_job(job_id: str) -> dict[str, Any]:
        find_job(job_id)
        if not isinstance(service, ProcessTranscriptionService):
            raise HTTPException(
                409, "Cancellation requires the process inference service"
            )
        return service.cancel(job_id).snapshot()

    return app


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="noteit-server")
    parser.add_argument("--host", default=os.environ.get("NOTEIT_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("NOTEIT_PORT", "8765"))
    )
    args = parser.parse_args()

    token = os.environ.get("NOTEIT_API_TOKEN")
    if not token:
        raise SystemExit(
            "NOTEIT_API_TOKEN is not set. Generate one with: "
            "python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )

    import uvicorn

    service = ProcessTranscriptionService(ServiceConfig.from_env())
    uvicorn.run(
        create_app(service, token),
        host=args.host,
        port=args.port,
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
