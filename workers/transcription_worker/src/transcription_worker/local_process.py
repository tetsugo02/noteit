"""Private JSON-lines worker protocol; credentials arrive on stdin, never argv."""

import contextlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from .models import DiarizerModelType, WhisperModelType
from .service import ServiceConfig, TranscriptionService
from .backends import FINISHED


def main():
    payload = json.loads(sys.stdin.readline())
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        options = payload["options"]
        audio = options.pop("audio_path")
        device = options.pop("device")
        options.pop("target", None)
        options["whisper_model"] = WhisperModelType(options["whisper_model"])
        options["diarizer_model"] = DiarizerModelType(options["diarizer_model"])
        directory = Path(payload["output_dir"])
        # Clean only noteit's private decoding files, including a previous crash.
        temporary = directory / "temporary"
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True, mode=0o700)
        import os

        os.environ["NOTEIT_TEMP_DIR"] = str(temporary)
        config = replace(
            ServiceConfig.from_env(),
            hf_token=payload["hf_token"],
            device=device,
            diarizer_device="auto",
            output_dir=directory,
        )
        service = TranscriptionService(config)
        job = service.submit(audio, job_id=payload["job_id"], **options)
        while True:
            print(json.dumps(job.snapshot()), file=protocol, flush=True)
            if job.status.value in FINISHED:
                break
            job.finished.wait(0.25)
        if job.result:
            print(
                json.dumps({"result": job.result.model_dump(mode="json")}),
                file=protocol,
                flush=True,
            )


if __name__ == "__main__":
    main()
