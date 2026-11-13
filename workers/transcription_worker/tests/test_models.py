import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transcription_worker.models import WhisperModelType, default_whisper_model


def test_catalog_and_defaults_do_not_import_inference_libraries():
    script = """
import json,sys
from transcription_worker.models import default_whisper_model, engine_capabilities
catalog=engine_capabilities()
default_whisper_model()
print(json.dumps({'devices':catalog['devices'],'loaded':[name for name in ['torch','ctranslate2','faster_whisper','mlx_whisper'] if name in sys.modules]}))
"""
    result = subprocess.check_output(
        [sys.executable, "-c", script],
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    state = json.loads(result)
    assert "cpu" in state["devices"]
    assert state["loaded"] == []


@pytest.mark.parametrize(
    "code,output,expected",
    [
        (0, "0\n", WhisperModelType.TURBO),
        (0, "", WhisperModelType.SMALL),
        (1, "", WhisperModelType.SMALL),
    ],
)
def test_cuda_defaults_require_successful_driver_probe(
    monkeypatch, code, output, expected
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    run = Mock(return_value=SimpleNamespace(returncode=code, stdout=output))
    monkeypatch.setattr(subprocess, "run", run)
    assert default_whisper_model() == expected
    assert run.call_args.kwargs["timeout"] == 2


def test_unresponsive_driver_probe_does_not_block_catalog(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("nvidia-smi", 2))
    )
    assert default_whisper_model() == WhisperModelType.SMALL
