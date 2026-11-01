"""MCP server that lets agents transcribe meeting recordings with speaker labels.

Transcription is slow on laptops, so it runs as a background job:

1. ``start_transcription`` returns a ``job_id`` immediately.
2. ``wait_for_transcription`` blocks up to ``wait_seconds`` and reports progress.
3. ``get_transcript`` returns ``[HH:MM:SS] SPEAKER_00: ...`` lines that the
   calling LLM can summarize. Speakers can be renamed via ``speaker_names``.

Run with ``noteit-mcp`` (stdio transport).
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from dotenv import load_dotenv
from mcp.server.mcpserver import Context, MCPServer
from pydantic import Field

from .models import (
    DIARIZER_MODEL_INFO,
    WHISPER_MODEL_INFO,
    DiarizerModelType,
    WhisperModelType,
)
from .service import JobStatus, ServiceConfig, TranscriptionService
from .transcript import format_timestamp, format_transcript

MAX_WAIT_SECONDS = 50

server = MCPServer(
    name="noteit",
    instructions=(
        "Transcribes local meeting recordings with speaker diarization "
        "(pyannote + Whisper). Call start_transcription, then call "
        "wait_for_transcription repeatedly until status is 'done', then "
        "get_transcript. Pick a lighter whisper_model (small/base) on machines "
        "without a GPU. Speakers are labelled SPEAKER_00, SPEAKER_01, ...; infer "
        "real names from context only when the transcript makes them clear."
    ),
)

_service: TranscriptionService | None = None


def get_service() -> TranscriptionService:
    global _service
    if _service is None:
        _service = TranscriptionService(ServiceConfig.from_env())
    return _service


@server.tool()
def list_models() -> dict[str, Any]:
    """List selectable Whisper / diarization models and the server defaults."""
    config = get_service().config
    return {
        "whisper_models": [
            {"name": m.value, "parameters": i.parameters, "note": i.note}
            for m, i in WHISPER_MODEL_INFO.items()
        ],
        "diarizer_models": [
            {"name": m.value, "note": i.note} for m, i in DIARIZER_MODEL_INFO.items()
        ],
        "defaults": {
            "whisper_model": config.whisper_model.value,
            "diarizer_model": config.diarizer_model.value,
            "device": config.device,
        },
    }


@server.tool()
def start_transcription(
    audio_path: Annotated[
        str,
        Field(
            description="Absolute path to a local audio/video file (wav, m4a, mp3, mp4, ...)."
        ),
    ],
    language: Annotated[
        str | None,
        Field(description="ISO 639-1 code such as 'ja' or 'en'. null = auto-detect."),
    ] = "ja",
    whisper_model: Annotated[
        WhisperModelType | None,
        Field(
            description="Recognition model size. null = server default. Lighter is faster."
        ),
    ] = None,
    diarizer_model: Annotated[
        DiarizerModelType | None,
        Field(description="Speaker diarization model. null = server default."),
    ] = None,
    num_speakers: Annotated[
        int | None,
        Field(
            ge=1, description="Exact number of speakers, if known. Improves accuracy."
        ),
    ] = None,
    min_speakers: Annotated[int | None, Field(ge=1)] = None,
    max_speakers: Annotated[int | None, Field(ge=1)] = None,
) -> dict[str, Any]:
    """Start transcribing a recording in the background and return its job_id."""
    job = get_service().submit(
        audio_path,
        language=language,
        whisper_model=whisper_model,
        diarizer_model=diarizer_model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    return job.snapshot()


@server.tool()
async def wait_for_transcription(
    job_id: str,
    ctx: Context,
    wait_seconds: Annotated[
        int,
        Field(
            ge=0,
            le=MAX_WAIT_SECONDS,
            description="How long to block waiting for completion.",
        ),
    ] = 30,
) -> dict[str, Any]:
    """Wait up to wait_seconds for a job, reporting progress, and return its status."""
    service = get_service()
    job = service.get_job(job_id)
    if job is None:
        raise ValueError(f"Unknown job_id: {job_id}")

    deadline = asyncio.get_running_loop().time() + wait_seconds
    while not job.finished.is_set():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        if job.segments_total:
            await ctx.report_progress(
                job.segments_done, job.segments_total, f"{job.status.value}"
            )
        await asyncio.to_thread(job.finished.wait, min(remaining, 2.0))
    return job.snapshot()


@server.tool()
def list_transcriptions() -> list[dict[str, Any]]:
    """List jobs in this session and transcripts saved by earlier sessions."""
    service = get_service()
    jobs = [job.snapshot() for job in service.list_jobs()]
    known = {job["job_id"] for job in jobs}
    saved = [
        {"job_id": path.stem, "status": JobStatus.DONE.value, "saved_to": str(path)}
        for path in service.list_saved_results()
        if path.stem not in known
    ]
    return jobs + saved


@server.tool()
def get_transcript(
    job_id: str,
    speaker_names: Annotated[
        dict[str, str] | None,
        Field(description='Optional rename map, e.g. {"SPEAKER_00": "Tanaka"}.'),
    ] = None,
) -> str:
    """Return the speaker-labelled transcript of a finished job as plain text."""
    result = get_service().get_result(job_id)
    duration = result.utterances[-1].end_time_seconds if result.utterances else 0.0
    header = (
        f"# Transcript {result.job_id}\n"
        f"audio: {result.audio_path}\n"
        f"duration: {format_timestamp(duration)} / speakers: {', '.join(result.speakers)} / "
        f"model: whisper-{result.whisper_model.value}, pyannote-{result.diarizer_model.value}\n"
    )
    return header + "\n" + format_transcript(result.utterances, speaker_names)


@server.prompt()
def summarize_meeting(job_id: str, speaker_names: str = "") -> str:
    """Summarize a finished transcription as Japanese meeting minutes."""
    names = dict(pair.split("=", 1) for pair in speaker_names.split(",") if "=" in pair)
    transcript = get_transcript(job_id, names or None)
    return (
        "以下は話者分離付きの会議の文字起こしです。音声認識の誤りが含まれる前提で、"
        "文脈から補って読んでください。\n\n"
        "次の形式で日本語の議事録を作成してください。\n"
        "1. 概要（3行以内）\n"
        "2. 議題ごとの議論内容（誰が何を主張したか）\n"
        "3. 決定事項\n"
        "4. TODO（担当者・期限が分かれば併記）\n"
        "5. 未解決の論点\n\n"
        f"{transcript}"
    )


def main() -> None:
    load_dotenv()
    server.run("stdio")


if __name__ == "__main__":
    main()
