import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
import threading

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config_loader import load_config
from engines.stt.audio_processor import convert_to_wav
from engines.stt.whisper_engine import WhisperEngine
from .ws_handler import stream_transcribe
from api.dashboard import router as dashboard_router
from api.devices import router as devices_router

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
assistant = None # Sẽ được set từ main.py khi khởi chạy tích hợp


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Đang khởi động API Server...")
    try:
        # Load engine nếu cấu hình whisper có sẵn
        whisper_cfg = CONFIG.get("whisper")
        if whisper_cfg:
            engine = WhisperEngine(config=whisper_cfg)
    except Exception as e:
        logger.error(f"Lỗi load Whisper engine: {e}")
    yield
    logger.info("API Server dừng.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Home Assistance API",
    description="API điều khiển Quản gia AI - STT & Control Hub",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dashboard_router)
app.include_router(devices_router)

# Mount static files (cho dashboard frontend)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", summary="Kiểm tra trạng thái server")
async def health():
    return {
        "status": "ok", 
        "model_ready": engine is not None, 
        "assistant_connected": assistant is not None
    }

# ─── Assistant Control Endpoints ─────────────────────────────────────────────

@app.get("/status", summary="Lấy trạng thái Quản gia AI")
async def get_status():
    if assistant is None:
        raise HTTPException(status_code=503, detail="Assistant chưa được kết nối")
    
    return {
        "is_unlocked": assistant.is_unlocked,
        "current_user": assistant.current_user,
        "is_authenticating": assistant.is_authenticating,
        "vision_active": assistant.vision.is_running if assistant.vision else False
    }


@app.post("/control", summary="Điều khiển hệ thống (lock/unlock)")
async def control_assistant(action: str = Query(..., description="lock hoặc unlock")):
    if assistant is None:
        raise HTTPException(status_code=503, detail="Assistant chưa được kết nối")
    
    action = action.lower()
    if action == "lock":
        # Khóa ngay lập tức
        assistant._lock_system()
        return {"message": "Đã khóa hệ thống"}
    elif action == "unlock":
        if not assistant.is_unlocked:
             # Logic unlock: Kích hoạt camera hoặc log request
             logger.info("API: Nhận yêu cầu mở khóa hệ thống")
             # assistant._handle_authentication() 
             return {"message": "Hệ thống đang sẵn sàng xử lý mở khóa"}
        return {"message": "Hệ thống đã mở khóa từ trước"}
    else:
        raise HTTPException(status_code=400, detail="Hành động không hợp lệ. Sử dụng 'lock' hoặc 'unlock'")


@app.post("/command", summary="Gửi lệnh văn bản cho AI")
async def send_command(text: str = Query(..., description="Nội dung lệnh thoại")):
    if assistant is None:
        raise HTTPException(status_code=503, detail="Assistant chưa được kết nối")
    
    logger.info(f"API Received Code: {text}")
    # Xử lý lệnh như một câu thoại thông qua Voice Engine
    # Sử dụng threading để không block event loop của FastAPI
    threading.Thread(target=assistant.voice.process_text_command, args=(text,), daemon=True).start()
    return {"message": f"Đã nhận lệnh: {text}"}


# ─── STT Endpoints ───────────────────────────────────────────────────────────

@app.post("/transcribe", summary="Nhận diện giọng nói từ file audio")
async def transcribe(audio: UploadFile = File(...)):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model STT chưa được load")

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


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    if engine is None:
        await websocket.close(code=1011, reason="Model STT chưa được load")
        return
    await stream_transcribe(websocket, engine)
