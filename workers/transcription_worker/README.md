# Transcription worker

The backend-facing entry point is the synchronous `TranscriptionWorker.transcribe`
method. Create the models once when the worker process starts and reuse the same
instance for every job.

```python
from pathlib import Path

from transcription_worker.models import DiarizerModelType, WhisperModelType
from transcription_worker.speaker_diarizer import SpeakerDiarizer
from transcription_worker.whisper import Whisper
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

## Setup

1. Accept the terms of `pyannote/speaker-diarization-community-1` on Hugging Face.
2. Put a token in `.env`: `HUGGINGFACE_ACCESS_TOKEN=hf_...`
3. `uv sync`

## Choosing a model size

| `whisper_model` | params | notes |
|---|---|---|
| `tiny` | 39M | fastest, rough Japanese |
| `base` | 74M | clear speech only |
| `small` | 244M | **default without CUDA** (MacBook) |
| `medium` | 769M | better, slower |
| `turbo` | 809M | **default with CUDA** |
| `large` | 1.55B | best, GPU recommended |

Whisper runs on CUDA, then Apple MPS, then CPU. pyannote uses CUDA or CPU
(MPS is opt-in via `NOTEIT_DIARIZER_DEVICE=mps`).

Environment variables: `NOTEIT_WHISPER_MODEL`, `NOTEIT_DIARIZER_MODEL`,
`NOTEIT_DEVICE`, `NOTEIT_DIARIZER_DEVICE`, `NOTEIT_OUTPUT_DIR`
(default `~/.cache/noteit/transcripts`), `WHISPER_COMPILE`.

## CLI

```sh
uv run transcribe-audio meeting.m4a -m small -n 3 -o transcript.txt
```

## MCP server (for Claude Code and other agents)

`noteit-mcp` serves over stdio. The repository's `.mcp.json` registers it for
Claude Code. For other clients:

```json
{"command": "uv", "args": ["run", "--directory", "/abs/path/workers/transcription_worker", "noteit-mcp"]}
```

Tools:

- `list_models`: available model sizes and defaults
- `start_transcription(audio_path, language="ja", whisper_model?, num_speakers?, ...)`: returns a `job_id`
- `wait_for_transcription(job_id, wait_seconds=30)`: blocks up to 50 s and reports progress
- `get_transcript(job_id, speaker_names?)`: returns `[HH:MM:SS] SPEAKER_00: ...` lines
- `list_transcriptions`: jobs in this session plus results saved earlier

Prompt: `summarize_meeting(job_id, speaker_names="SPEAKER_00=田中,...")` builds
Japanese meeting minutes from the transcript. The calling LLM does the summarizing,
so the server needs no LLM API key.

Transcription runs as a background job because a long meeting can take many minutes
on a laptop, which is longer than an MCP client's tool timeout.

Note: transformers and pyannote normally decode files with `torchcodec`, which
fails with FFmpeg 9 (current Homebrew). This package decodes audio with pydub and
libsndfile and passes arrays to both models, so any installed FFmpeg works.
