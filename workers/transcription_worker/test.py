import os
from itertools import groupby
from pathlib import Path

from dotenv import load_dotenv
from transcription_worker.schema import SpeakerAssignedWord
from transcription_worker.speaker_diarizer import DiarizerModelType, SpeakerDiarizer
from transcription_worker.whisper import Whisper, WhisperModelType
from transcription_worker.worker import TranscriptionWorker


def print_result(words: list[SpeakerAssignedWord]) -> None:
    """Print both a readable transcript and the schema-compatible JSON result."""
    print("\n=== Transcript by speaker ===")
    for speaker_label, grouped_words in groupby(
        words, key=lambda word: word.speaker_label
    ):
        speaker_words = list(grouped_words)
        start = speaker_words[0].start_time_seconds
        end = speaker_words[-1].end_time_seconds
        text = "".join(word.word for word in speaker_words)
        print(f"[{start:7.2f}s - {end:7.2f}s] {speaker_label}: {text}")

    print("\n=== Worker schema JSON ===")
    print("[\n" + ",\n".join(word.model_dump_json() for word in words) + "\n]")


def main() -> None:
    load_dotenv()
    token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    if not token:
        raise ValueError(
            "HUGGINGFACE_ACCESS_TOKEN is not set in the environment variables."
        )

    audio_file_path = Path(os.environ.get("SAMPLE_AUDIO_FILE", "./jojo.wav"))
    if not audio_file_path.is_file():
        raise FileNotFoundError(f"Sample audio file not found: {audio_file_path}")

    diarizer = SpeakerDiarizer(hf_token=token, model_type=DiarizerModelType.COMMUNITY)
    compile_model = os.environ.get("WHISPER_COMPILE", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    whisper = Whisper(
        hf_token=token,
        model_type=WhisperModelType.TURBO,
        compile_model=compile_model,
    )
    worker = TranscriptionWorker(diarizer=diarizer, whisper=whisper)
    words = worker.transcribe(
        audio_file_path,
        language=os.environ.get("WHISPER_LANGUAGE", "ja"),
        on_segment=lambda segment: print(
            f"Transcribing {segment.speaker_label}: "
            f"{segment.start_time:.2f}s - {segment.end_time:.2f}s"
        ),
    )
    print_result(words)


if __name__ == "__main__":
    main()
