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

from .backends import FINISHED, TranscriptionBackend
from .models import DiarizerModelType, WhisperModelType
from .transcript import format_timestamp, format_transcript

MAX_WAIT_SECONDS = 50

server = MCPServer(
    name="noteit",
    instructions=(
        "Transcribes local meeting recordings with optional speaker diarization "
        "(pyannote + Whisper). Call start_transcription, then call "
        "wait_for_transcription repeatedly until status is 'done', then "
        "get_transcript. Pick a lighter whisper_model (small/base) on machines "
        "without a GPU. Speakers are labelled SPEAKER_00, SPEAKER_01, ...; infer "
        "real names from context only when the transcript makes them clear."
    ),
)

_backend: TranscriptionBackend | None = None


def get_backend() -> TranscriptionBackend:
    """Connect to the shared desktop service, starting it lazily if necessary."""
    global _backend
    if _backend is None:
        from .frontend_client import FrontendBackend

        _backend = FrontendBackend()
    return _backend


@server.tool()
def list_models() -> dict[str, Any]:
    """List selectable Whisper / diarization models, defaults, and where they run."""
    return get_backend().describe()


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
        Field(
            description="ISO code such as ja/en; auto = detect; null/omitted = shared setting."
        ),
    ] = None,
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
    diarize: bool = False,
    target: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Start transcribing a recording in the background and return its job_id."""
    return get_backend().submit(
        audio_path,
        language=language,
        whisper_model=whisper_model,
        diarizer_model=diarizer_model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        diarize=diarize,
        **({"target": target} if target else {}),
        **({"device": device} if device else {}),
    )


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
    backend = get_backend()
    deadline = asyncio.get_running_loop().time() + wait_seconds
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        snapshot = await asyncio.to_thread(
            backend.wait, job_id, max(0.0, min(remaining, 5.0))
        )
        if snapshot.get("segments_total"):
            await ctx.report_progress(
                snapshot["segments_done"],
                snapshot["segments_total"],
                snapshot["status"],
            )
        if snapshot["status"] in FINISHED or remaining <= 5.0:
            return snapshot


@server.tool()
def list_transcriptions() -> list[dict[str, Any]]:
    """List jobs in this session and transcripts saved by earlier sessions."""
    return get_backend().list_jobs()


@server.tool()
def get_transcript(
    job_id: str,
    speaker_names: Annotated[
        dict[str, str] | None,
        Field(description='Optional rename map, e.g. {"SPEAKER_00": "Tanaka"}.'),
    ] = None,
) -> str:
    """Return the speaker-labelled transcript of a finished job as plain text."""
    result = get_backend().get_result(job_id)
    duration = result.utterances[-1].end_time_seconds if result.utterances else 0.0
    diarizer = (
        f", pyannote-{result.diarizer_model.value}" if result.diarizer_model else ""
    )
    header = (
        f"# Transcript {result.job_id}\n"
        f"audio: {result.audio_path}\n"
        f"duration: {format_timestamp(duration)} / speakers: {', '.join(result.speakers)} / "
        f"model: whisper-{result.whisper_model.value}{diarizer}\n"
    )
    return header + "\n" + format_transcript(result.utterances, speaker_names)


@server.tool()
def get_transcription_settings() -> dict:
    """Read the same settings and remote profiles used by the desktop GUI."""
    from .frontend_client import FrontendBackend

    backend = get_backend()
    if not isinstance(backend, FrontendBackend):
        return {"settings": {}, "profiles": []}
    return {
        "settings": backend.request("GET", "/settings"),
        "profiles": backend.request("GET", "/profiles"),
    }


@server.tool()
def update_transcription_settings(
    target: str | None = None,
    device: str | None = None,
    whisper_model: WhisperModelType | None = None,
    language: str | None = None,
    diarize: bool | None = None,
) -> dict:
    """Update desktop defaults. Existing jobs retain their original execution target."""
    backend = get_backend()
    current = backend.request("GET", "/settings")
    updates = dict(
        target=target,
        device=device,
        whisper_model=whisper_model,
        language=language,
        diarize=diarize,
    )
    current.update({key: value for key, value in updates.items() if value is not None})
    return backend.request("PUT", "/settings", json=current)


@server.prompt()
def summarize_meeting(job_id: str, speaker_names: str = "") -> str:
    """Summarize a finished transcription as Japanese meeting minutes."""
    names = dict(pair.split("=", 1) for pair in speaker_names.split(",") if "=" in pair)
    transcript = get_transcript(job_id, names or None)
    return (
        "以下は話者分離付きの会議の文字起こしです．音声認識の誤りが含まれます．"
        "不明瞭な専門用語や発言内容を推測で確定せず，要確認と明示してください．"
        "SPEAKER_UNKNOWNは話者不明として扱い，人物名を推測しないでください．"
        "決定事項・TODO・担当者・期限は発言に根拠があるものだけ記載し，"
        "不明な項目は不明としてください．\n\n"
        "次の形式で日本語の議事録を作成してください．\n"
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
