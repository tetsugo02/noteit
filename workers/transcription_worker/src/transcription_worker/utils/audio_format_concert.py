from pathlib import Path

from pydub import AudioSegment


def convert_audio_format(input_file: Path, output_file: Path) -> None:
    input_format = input_file.suffix.lower()
    output_format = output_file.suffix.lower()

    try:
        audio: AudioSegment = AudioSegment.from_file(
            input_file, format=input_format[1:]
        )
        audio.export(output_file, format=output_format[1:])
        print(f"Converted {input_file} to {output_file}")
    except ImportError:
        print(
            "pydub is not installed. Please install it to use the audio conversion feature."
        )
    except Exception as e:
        print(f"Error converting audio: {e}")
