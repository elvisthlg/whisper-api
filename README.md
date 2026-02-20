# whisper.cpp FastAPI service

This project exposes a FastAPI API that wraps `whisper.cpp` and protects requests with a bearer token.

## Features

- Accepts uploaded audio files with `multipart/form-data`
- Converts incoming audio to whisper-friendly WAV format (16kHz, mono, PCM s16le)
- Enqueues transcription jobs in an internal queue (single worker)
- Runs `whisper.cpp` CLI and returns transcribed text
- Uses bearer token authentication (`Authorization: Bearer <token>`)

## Environment variables

- `API_TOKEN` **(required)**: bearer token expected by the API
- `WHISPER_CPP_BIN` (default: `whisper-cli`): path to whisper.cpp executable
- `WHISPER_MODEL_PATH` (default: `models/ggml-base.en.bin`): model file path
- `FFMPEG_BIN` (default: `ffmpeg`): ffmpeg binary used for audio conversion
- `TRANSCRIBE_TIMEOUT_SECONDS` (default: `600`): request wait timeout

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export API_TOKEN="change-me"
export WHISPER_CPP_BIN="/path/to/whisper-cli"
export WHISPER_MODEL_PATH="/path/to/model.ggml.bin"
export FFMPEG_BIN="/usr/bin/ffmpeg"

uvicorn main:app --host 0.0.0.0 --port 8000
```

## Request example

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "Authorization: Bearer change-me" \
  -F "file=@sample.mp3" \
  -F "language=en"
```

Response:

```json
{
  "task_id": "6f032f1b-03f5-43ca-8139-c4e9228e4de9",
  "text": "your transcription text"
}
```

## Docker (VPS deployment)

Build image:

```bash
docker build -t whisper-api:latest .
```

Run container (mount your model and set API token):

```bash
docker run -d --name whisper-api \
  -p 8000:8000 \
  -e API_TOKEN="change-me" \
  -e WHISPER_MODEL_PATH="/models/ggml-base.en.bin" \
  -v /opt/whisper-models:/models:ro \
  whisper-api:latest
```

The image builds `whisper-cli` from source and installs `ffmpeg`, so audio conversion + transcription works out-of-the-box.

