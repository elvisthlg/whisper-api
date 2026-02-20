# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS whisper-builder

ARG WHISPER_CPP_REPO=https://github.com/ggerganov/whisper.cpp.git
ARG WHISPER_CPP_REF=master

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone --depth 1 --branch "${WHISPER_CPP_REF}" "${WHISPER_CPP_REPO}" whisper.cpp
WORKDIR /tmp/whisper.cpp
RUN cmake -S . -B build -DWHISPER_BUILD_TESTS=OFF \
    && cmake --build build --config Release -j"$(nproc)"


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WHISPER_CPP_BIN=/usr/local/bin/whisper-cli \
    WHISPER_MODEL_PATH=/models/ggml-base.en.bin \
    TRANSCRIBE_TIMEOUT_SECONDS=600

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=whisper-builder /tmp/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli
COPY main.py ./

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /models \
    && chown -R appuser:appuser /app /models

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
