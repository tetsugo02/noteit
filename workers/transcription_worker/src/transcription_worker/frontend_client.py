"""Shared desktop service client used by stdio MCP."""

import time
from typing import Any

import httpx

from .service import TranscriptionResult


class FrontendBackend:
    def __init__(self, endpoint=None):
        if endpoint is None:
            from .frontend import ensure_service

            endpoint = ensure_service()
        self.client = httpx.Client(
            base_url=endpoint["url"],
            headers={"Authorization": "Bearer " + endpoint["token"]},
            timeout=30,
        )

    def request(self, method, path, **kwargs):
        response = self.client.request(method, path, **kwargs)
        if response.status_code >= 400:
            if response.status_code == 404:
                raise KeyError(response.json().get("detail"))
            raise RuntimeError(response.json().get("detail", response.text))
        return response.json()

    def describe(self):
        return self.request("GET", "/models")

    def submit(self, audio_path, **options: Any):
        settings = self.request("GET", "/settings")
        options = {key: value for key, value in options.items() if value is not None}
        if options.get("language") == "auto":
            options["language"] = None
        return self.request(
            "POST", "/jobs", json={**settings, **options, "audio_path": str(audio_path)}
        )

    def wait(self, job_id, timeout):
        from .backends import FINISHED

        deadline = time.monotonic() + timeout
        while True:
            job = self.request("GET", f"/jobs/{job_id}")
            if job["status"] in FINISHED or time.monotonic() >= deadline:
                return job
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))

    def get_result(self, job_id):
        return TranscriptionResult.model_validate(
            self.request("GET", f"/jobs/{job_id}/result")
        )

    def list_jobs(self):
        return self.request("GET", "/jobs")
