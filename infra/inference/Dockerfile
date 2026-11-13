# Build with --platform=linux/amd64; Compose selects this automatically.
FROM python:3.12.12-slim-bookworm AS cpu
ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy NOTEIT_HOST=0.0.0.0 NOTEIT_PORT=8765 NOTEIT_DEVICE=cpu NOTEIT_DIARIZER_DEVICE=cpu NOTEIT_OUTPUT_DIR=/data/transcripts HF_HOME=/data/models NOTEIT_TEMP_DIR=/data/temporary
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv
WORKDIR /app
COPY workers/transcription_worker/pyproject.toml workers/transcription_worker/uv.lock ./
COPY workers/transcription_worker/src ./src
COPY workers/transcription_worker/README.md ./README.md
RUN uv sync --frozen --no-dev --extra server && useradd --create-home --uid 10001 noteit && mkdir /data && chown noteit:noteit /data
USER noteit
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3)"]
CMD ["/app/.venv/bin/noteit-server"]
