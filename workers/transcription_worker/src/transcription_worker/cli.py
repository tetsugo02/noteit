"""Command-line transcription: ``transcribe-audio meeting.m4a --whisper-model small``."""

from __future__ import annotations

import argparse
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
    parser.add_argument("-n", "--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument(
        "-o", "--output", type=Path, help="Write the transcript text here"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    config = ServiceConfig.from_env()
    if args.device:
        config = replace(config, device=args.device)
    service = TranscriptionService(config)

    job = service.submit(
        args.audio,
        language=None if args.language == "auto" else args.language,
        whisper_model=args.whisper_model,
        diarizer_model=args.diarizer_model,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    print(
        f"job {job.id}: whisper-{job.whisper_model}, pyannote-{job.diarizer_model}",
        file=sys.stderr,
    )

    last = None
    while not job.finished.wait(1.0):
        state = (job.status, job.segments_done, job.segments_total)
        if state != last:
            progress = (
                f" {job.segments_done}/{job.segments_total}"
                if job.segments_total
                else ""
            )
            print(
                f"\r{job.status.value}{progress}   ",
                end="",
                file=sys.stderr,
                flush=True,
            )
            last = state
    print(file=sys.stderr)

    if job.status is JobStatus.FAILED:
        print(f"failed: {job.error}", file=sys.stderr)
        return 1

    result = service.get_result(job.id)
    text = format_transcript(result.utterances)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    print(
        f"done in {result.elapsed_seconds:.0f}s; JSON saved to "
        f"{config.output_dir / (job.id + '.json')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
