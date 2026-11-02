"""Command-line transcription: ``transcribe-audio meeting.m4a --whisper-model small``."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from .models import (
    DEVICE_CHOICES,
    WHISPER_MODEL_INFO,
    DiarizerModelType,
    WhisperModelType,
)
from .backends import FINISHED, LocalBackend, RemoteBackend, TranscriptionBackend
from .service import JobStatus, ServiceConfig, TranscriptionService
from .transcript import format_transcript


def build_parser() -> argparse.ArgumentParser:
    models_help = "\n".join(
        f"  {m.value:<7} {i.parameters:>6}  {i.note}"
        for m, i in WHISPER_MODEL_INFO.items()
    )
    parser = argparse.ArgumentParser(
        prog="transcribe-audio",
        description="Speaker-labelled transcription with pyannote + Whisper.",
        epilog=f"whisper models:\n{models_help}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "-m", "--whisper-model", type=WhisperModelType, choices=list(WhisperModelType)
    )
    parser.add_argument(
        "--diarizer-model", type=DiarizerModelType, choices=list(DiarizerModelType)
    )
    parser.add_argument(
        "-l", "--language", default="ja", help="'auto' to detect (default: ja)"
    )
    parser.add_argument(
        "--device", choices=DEVICE_CHOICES, help="Whisper device (default: auto)"
    )
    parser.add_argument(
        "--no-diarize", action="store_true", help="Whisper only; no HF token required"
    )
    parser.add_argument("-n", "--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument(
        "--server",
        default=os.environ.get("NOTEIT_SERVER_URL"),
        help="Inference server URL (default: $NOTEIT_SERVER_URL; unset = run locally)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Write the transcript text here"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    backend: TranscriptionBackend
    if args.server:
        backend = RemoteBackend(args.server, os.environ.get("NOTEIT_API_TOKEN"))
    else:
        config = ServiceConfig.from_env()
        if args.device:
            config = replace(config, device=args.device)
        backend = LocalBackend(TranscriptionService(config))

    snapshot = backend.submit(
        args.audio,
        language=None if args.language == "auto" else args.language,
        whisper_model=args.whisper_model,
        diarizer_model=args.diarizer_model,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        diarize=not args.no_diarize,
    )
    job_id = snapshot["job_id"]
    where = args.server or "local"
    print(
        f"job {job_id} ({where}): whisper-{snapshot['whisper_model']}"
        + (
            f", pyannote-{snapshot['diarizer_model']}"
            if snapshot["diarizer_model"]
            else ""
        ),
        file=sys.stderr,
    )

    last = None
    while snapshot["status"] not in FINISHED:
        snapshot = backend.wait(job_id, 5)
        state = (
            snapshot["status"],
            snapshot["segments_done"],
            snapshot["segments_total"],
        )
        if state != last:
            total = snapshot["segments_total"]
            progress = f" {snapshot['segments_done']}/{total}" if total else ""
            print(
                f"\r{snapshot['status']}{progress}   ",
                end="",
                file=sys.stderr,
                flush=True,
            )
            last = state
    print(file=sys.stderr)

    if snapshot["status"] in {JobStatus.FAILED.value, JobStatus.INTERRUPTED.value}:
        print(f"failed: {snapshot['error']}", file=sys.stderr)
        return 1

    result = backend.get_result(job_id)
    text = format_transcript(result.utterances)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    print(f"done in {result.elapsed_seconds:.0f}s (job {job_id})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
