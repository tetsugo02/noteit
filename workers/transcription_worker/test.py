import os
from pathlib import Path

from dotenv import load_dotenv
from transcription_worker.speaker_diarizer import DiarizerModelType, SpeakerDiarizer
from transcription_worker.utils.audio_format_concert import convert_audio_format


def main() -> None:
    load_dotenv()
    token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    if not token:
        raise ValueError(
            "HUGGINGFACE_ACCESS_TOKEN is not set in the environment variables."
        )

    audio_file_path = Path("./jojo.wav")

    diarizer = SpeakerDiarizer(hf_token=token, model_type=DiarizerModelType.COMMUNITY)
    output = diarizer.diarize(audio_file_path)
    print("Diarization Output:", output)


def convert_audio() -> None:
    input_file = "./jojo.m4a"
    output_file = "./jojo.wav"

    convert_audio_format(Path(input_file), Path(output_file))


if __name__ == "__main__":
    main()
