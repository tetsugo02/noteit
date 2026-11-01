import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from mcp import Client
from pydub import AudioSegment
from transcription_worker import mcp_server
from transcription_worker.models import DiarizerModelType, WhisperModelType
from transcription_worker.schema import SpeakerAssignedWord, SpeakerSegment
from transcription_worker.service import (
    JobStatus,
    ServiceConfig,
    TranscriptionService,
)


class FakeWorker:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, audio_path, *, on_diarized, on_segment, **kwargs):
        self.calls.append({"audio_path": audio_path, **kwargs})
        segments = [
            SpeakerSegment(speaker_label="SPEAKER_00", start_time=0.0, end_time=1.0),
            SpeakerSegment(speaker_label="SPEAKER_01", start_time=1.0, end_time=2.0),
        ]
        on_diarized(segments)
        for segment in segments:
            on_segment(segment)
        return [
            SpeakerAssignedWord(
                word=text,
                start_time_seconds=start,
                end_time_seconds=start + 0.5,
                probablility=None,
                speaker_label=speaker,
            )
            for text, start, speaker in [
                ("よろしく", 0.0, "SPEAKER_00"),
                ("お願いします", 0.5, "SPEAKER_00"),
                ("はい", 1.2, "SPEAKER_01"),
            ]
        ]


@pytest.fixture
def audio_path(tmp_path: Path) -> Path:
    path = tmp_path / "meeting.wav"
    AudioSegment.silent(duration=2000).export(path, format="wav")
    return path


@pytest.fixture
def service(tmp_path: Path) -> tuple[TranscriptionService, FakeWorker, Mock]:
    config = ServiceConfig(
        hf_token="token",
        whisper_model=WhisperModelType.SMALL,
        diarizer_model=DiarizerModelType.COMMUNITY,
        device="cpu",
        diarizer_device="cpu",
        compile_whisper=False,
        output_dir=tmp_path / "out",
    )
    svc = TranscriptionService(config)
    worker = FakeWorker()
    loader = Mock(return_value=worker)
    svc._load_worker = loader  # type: ignore[method-assign]
    return svc, worker, loader


def test_job_runs_with_requested_model_and_persists_result(service, audio_path) -> None:
    svc, worker, loader = service

    job = svc.submit(audio_path, whisper_model=WhisperModelType.BASE, num_speakers=2)
    svc.wait(job.id, timeout=10)

    assert job.status is JobStatus.DONE, job.error
    assert (job.segments_done, job.segments_total) == (2, 2)
    loader.assert_called_once_with(WhisperModelType.BASE, DiarizerModelType.COMMUNITY)
    assert worker.calls[0]["num_speakers"] == 2

    result = svc.get_result(job.id)
    assert [u.text for u in result.utterances] == ["よろしくお願いします", "はい"]
    assert result.speakers == ["SPEAKER_00", "SPEAKER_01"]

    saved = json.loads((svc.config.output_dir / f"{job.id}.json").read_text())
    assert saved["whisper_model"] == "base"

    # A fresh service (e.g. after restart) can still read the saved result.
    fresh = TranscriptionService(svc.config)
    assert fresh.get_result(job.id).utterances == result.utterances


def test_failed_job_reports_error(service, audio_path) -> None:
    svc, _, loader = service
    loader.side_effect = RuntimeError("no token")

    job = svc.submit(audio_path)
    svc.wait(job.id, timeout=10)

    assert job.status is JobStatus.FAILED
    assert "no token" in (job.error or "")
    with pytest.raises(RuntimeError, match="failed"):
        svc.get_result(job.id)


def test_submit_rejects_missing_file(service, tmp_path) -> None:
    svc, _, _ = service
    with pytest.raises(FileNotFoundError):
        svc.submit(tmp_path / "missing.wav")


def test_get_result_rejects_path_like_job_ids(service) -> None:
    svc, _, _ = service
    with pytest.raises(KeyError):
        svc.get_result("../../etc/passwd")


@pytest.mark.anyio
async def test_mcp_tools_end_to_end(service, audio_path, monkeypatch) -> None:
    svc, _, _ = service
    monkeypatch.setattr(mcp_server, "_service", svc)

    async with Client(mcp_server.server) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
        assert {
            "list_models",
            "start_transcription",
            "wait_for_transcription",
            "get_transcript",
            "list_transcriptions",
        } <= tools

        started = await client.call_tool(
            "start_transcription",
            {"audio_path": str(audio_path), "whisper_model": "tiny"},
        )
        assert not started.is_error, started.content
        job_id = started.structured_content["job_id"]

        status = await client.call_tool(
            "wait_for_transcription", {"job_id": job_id, "wait_seconds": 10}
        )
        assert status.structured_content["status"] == "done"

        transcript = await client.call_tool(
            "get_transcript",
            {"job_id": job_id, "speaker_names": {"SPEAKER_00": "田中"}},
        )
        text = transcript.content[0].text
        assert "[00:00:00] 田中: よろしくお願いします" in text
        assert "[00:00:01] SPEAKER_01: はい" in text

        prompt = await client.get_prompt(
            "summarize_meeting", {"job_id": job_id, "speaker_names": "SPEAKER_01=鈴木"}
        )
        assert "鈴木: はい" in prompt.messages[0].content.text
