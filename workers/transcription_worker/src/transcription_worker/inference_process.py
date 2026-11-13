"""One HTTP inference job. JSON-lines stdout is reserved for the parent queue."""

import contextlib
import json
from pathlib import Path
import sys

from .models import DiarizerModelType, WhisperModelType
from .service import ServiceConfig, TranscriptionService


def main():
    payload = json.loads(sys.stdin.readline())
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        config = payload["config"]
        config["output_dir"] = Path(config["output_dir"])
        config["whisper_model"] = WhisperModelType(config["whisper_model"])
        config["diarizer_model"] = DiarizerModelType(config["diarizer_model"])
        options = payload["options"]
        options["whisper_model"] = WhisperModelType(options["whisper_model"])
        if options["diarizer_model"] is not None:
            options["diarizer_model"] = DiarizerModelType(options["diarizer_model"])
        # This private directory has its own transient SQLite DB. Opening the
        # parent's DB here would incorrectly recover its active/queued jobs.
        service = TranscriptionService(ServiceConfig(**config))
        try:
            job = service.submit(payload["audio_path"], **options)
            while not job.finished.wait(0.25):
                print(json.dumps(job.snapshot()), file=protocol, flush=True)
            print(json.dumps(job.snapshot()), file=protocol, flush=True)
            if job.result:
                print(
                    json.dumps({"result": job.result.model_dump(mode="json")}),
                    file=protocol,
                    flush=True,
                )
        finally:
            service.close()


if __name__ == "__main__":
    main()
