import asyncio
import contextlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel


WHISPER_CPP_BIN = os.getenv("WHISPER_CPP_BIN", "whisper-cli")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "models/ggml-base.en.bin")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
API_TOKEN = os.getenv("API_TOKEN", "")
TRANSCRIBE_TIMEOUT_SECONDS = int(os.getenv("TRANSCRIBE_TIMEOUT_SECONDS", "600"))


@dataclass
class TranscriptionTask:
    task_id: str
    audio_path: Path
    language: Optional[str]
    prompt: Optional[str]
    future: asyncio.Future[str]


class TranscriptionResponse(BaseModel):
    task_id: str
    text: str


app = FastAPI(title="whisper.cpp API", version="1.1.0")
app.state.queue = asyncio.Queue()
app.state.worker_task = None


async def verify_token(authorization: str = Header(default="")) -> None:
    expected = f"Bearer {API_TOKEN}"
    if not API_TOKEN or authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


async def convert_audio_to_whisper_format(source_path: Path) -> Path:
    """Normalize audio to 16kHz mono WAV before transcription."""
    normalized_path = source_path.with_suffix(".normalized.wav")
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(normalized_path),
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0 or not normalized_path.exists():
        raise RuntimeError(
            "audio conversion failed: "
            f"{stderr.decode().strip() or stdout.decode().strip() or 'unknown ffmpeg error'}"
        )

    return normalized_path


async def run_whisper(task: TranscriptionTask) -> str:
    with tempfile.TemporaryDirectory(prefix=f"whisper-{task.task_id}-") as tmpdir:
        output_prefix = str(Path(tmpdir) / "result")
        cmd = [
            WHISPER_CPP_BIN,
            "-m",
            WHISPER_MODEL_PATH,
            "-f",
            str(task.audio_path),
            "-oj",
            "-of",
            output_prefix,
        ]

        if task.language:
            cmd.extend(["-l", task.language])
        if task.prompt:
            cmd.extend(["--prompt", task.prompt])

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(
                "whisper.cpp failed with code "
                f"{process.returncode}: {stderr.decode().strip() or stdout.decode().strip()}"
            )

        output_file = Path(f"{output_prefix}.json")
        if not output_file.exists():
            raise RuntimeError("whisper.cpp did not create JSON output")

        payload = json.loads(output_file.read_text(encoding="utf-8"))
        text = payload.get("text")
        if not isinstance(text, str):
            raise RuntimeError("JSON output missing transcription text")

        return text.strip()


async def worker() -> None:
    queue: asyncio.Queue[TranscriptionTask] = app.state.queue
    while True:
        task = await queue.get()
        try:
            text = await run_whisper(task)
            if not task.future.done():
                task.future.set_result(text)
        except Exception as exc:  # noqa: BLE001
            if not task.future.done():
                task.future.set_exception(exc)
        finally:
            task.audio_path.unlink(missing_ok=True)
            queue.task_done()


@app.on_event("startup")
async def startup_event() -> None:
    if not API_TOKEN:
        raise RuntimeError("API_TOKEN environment variable is required")
    if app.state.worker_task is None:
        app.state.worker_task = asyncio.create_task(worker())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    worker_task = app.state.worker_task
    if worker_task:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    _: None = Depends(verify_token),
    file: UploadFile = File(...),
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> TranscriptionResponse:
    task_id = str(uuid.uuid4())
    source_suffix = Path(file.filename or "audio").suffix or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=source_suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        source_audio_path = Path(tmp.name)

    try:
        normalized_audio_path = await convert_audio_to_whisper_format(source_audio_path)
    finally:
        source_audio_path.unlink(missing_ok=True)

    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    task = TranscriptionTask(
        task_id=task_id,
        audio_path=normalized_audio_path,
        language=language,
        prompt=prompt,
        future=future,
    )
    await app.state.queue.put(task)

    try:
        text = await asyncio.wait_for(future, timeout=TRANSCRIBE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Transcription timed out") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TranscriptionResponse(task_id=task_id, text=text)
