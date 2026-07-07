# Transcription worker

The backend-facing entry point is the synchronous `TranscriptionWorker.transcribe`
method. Create the models once when the worker process starts and reuse the same
instance for every job.

```python
from pathlib import Path

from transcription_worker.speaker_diarizer import DiarizerModelType, SpeakerDiarizer
from transcription_worker.whisper import Whisper, WhisperModelType
from transcription_worker.worker import TranscriptionWorker

diarizer = SpeakerDiarizer(
    hf_token=hf_token,
    model_type=DiarizerModelType.COMMUNITY,
)
whisper = Whisper(
    hf_token=hf_token,
    model_type=WhisperModelType.TURBO,
    compile_model=True,
)
worker = TranscriptionWorker(diarizer=diarizer, whisper=whisper)

# Call this from a queue consumer. The return value is a list of
# SpeakerAssignedWord Pydantic models and can be serialized with model_dump().
words = worker.transcribe(Path("audio.wav"), language="ja")
payload = [word.model_dump(mode="json") for word in words]
```

`transcribe` is blocking and serializes access to the model. An async HTTP backend
should enqueue a job and let the worker process call this method instead of running
GPU inference directly on the API event loop.
