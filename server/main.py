import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config_loader import load_config
from engines.stt.audio_processor import convert_to_wav
from engines.stt.whisper_engine import WhisperEngine
from .ws_handler import stream_transcribe

# ─── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Load config ──────────────────────────────────────────────────────────────
CONFIG = load_config()

# ─── Khởi tạo engine (load 1 lần lúc startup) ────────────────────────────────
engine: WhisperEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Đang khởi động STT Server...")
    engine = WhisperEngine(config=CONFIG.get("whisper"))
    yield
    logger.info("STT Server dừng.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Home Assistance STT API",
    description="Nhận diện giọng nói tiếng Việt offline - Powered by Faster-Whisper",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", summary="Kiểm tra trạng thái server")
async def health():
    return {"status": "ok", "model_ready": engine is not None}


@app.post("/transcribe", summary="Nhận diện giọng nói từ file audio")
async def transcribe(audio: UploadFile = File(...)):
    """
    Upload file audio (wav, mp3, ogg, m4a, ...) và nhận về văn bản tiếng Việt.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    content = await audio.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File audio rỗng")

    try:
        start = time.perf_counter()
        audio_array = convert_to_wav(content, filename=audio.filename)
        result = engine.transcribe(audio_array)
        elapsed = round(time.perf_counter() - start, 3)

        return JSONResponse({
            "success": True,
            "processing_time_sec": elapsed,
            **result,
        })
    except Exception as e:
        logger.error(f"Lỗi transcribe: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/bytes", summary="Nhận diện từ raw bytes PCM float32")
async def transcribe_bytes(audio: UploadFile = File(...)):
    """
    Nhận raw PCM float32 16kHz mono bytes, trả về text.
    Dùng khi client đã có numpy array và muốn gửi trực tiếp.
    """
    import numpy as np

    if engine is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    content = await audio.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Dữ liệu rỗng")

    try:
        start = time.perf_counter()
        audio_array = np.frombuffer(content, dtype=np.float32)
        result = engine.transcribe(audio_array)
        elapsed = round(time.perf_counter() - start, 3)

        return JSONResponse({
            "success": True,
            "processing_time_sec": elapsed,
            **result,
        })
    except Exception as e:
        logger.error(f"Lỗi transcribe_bytes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint nhận audio stream PCM float32 realtime.
    Client gửi bytes từng chunk, gửi b'END' để kết thúc.
    """
    if engine is None:
        await websocket.close(code=1011, reason="Model chưa sẵn sàng")
        return
    await stream_transcribe(websocket, engine)
