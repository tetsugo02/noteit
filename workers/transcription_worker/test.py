import os
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from pydub import AudioSegment
from transcription_worker.schema import SpeakerAssignedWord
from transcription_worker.speaker_diarizer import DiarizerModelType, SpeakerDiarizer
from transcription_worker.whisper import Whisper, WhisperModelType


def diarize_and_transcribe(
    audio_file_path: Path,
    diarizer: SpeakerDiarizer,
    whisper: Whisper,
    *,
    language: str | None = None,
) -> list[SpeakerAssignedWord]:
    """Split audio by diarization segments and transcribe each segment."""
    speaker_segments = diarizer.diarize(audio_file_path)
    audio = AudioSegment.from_wav(audio_file_path)
    result: list[SpeakerAssignedWord] = []

    with TemporaryDirectory(prefix="noteit-transcription-") as temporary_directory:
        temporary_path = Path(temporary_directory)

        for index, segment in enumerate(speaker_segments):
            start_ms = max(0, round(segment.start_time * 1000))
            end_ms = min(len(audio), round(segment.end_time * 1000))
            if end_ms <= start_ms:
                continue

            segment_path = temporary_path / f"segment-{index:04d}.wav"
            audio[start_ms:end_ms].export(segment_path, format="wav")

            print(
                f"Transcribing {segment.speaker_label}: "
                f"{segment.start_time:.2f}s - {segment.end_time:.2f}s"
            )
            words = whisper.transcribe(segment_path, language=language)
            for word in words:
                result.append(
                    SpeakerAssignedWord(
                        word=word.word,
                        start_time_seconds=segment.start_time
                        + word.start_time_seconds,
                        end_time_seconds=segment.start_time + word.end_time_seconds,
                        probablility=word.probablility,
                        speaker_label=segment.speaker_label,
                    )
                )

    return result


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
    words = diarize_and_transcribe(
        audio_file_path,
        diarizer,
        whisper,
        language=os.environ.get("WHISPER_LANGUAGE", "ja"),
    )
    print_result(words)


if __name__ == "__main__":
    main()
