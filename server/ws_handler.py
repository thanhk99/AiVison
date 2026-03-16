import asyncio
import logging
from typing import AsyncGenerator

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from engines.stt.audio_processor import SAMPLE_RATE
from engines.stt.whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)

# Buffer nhỏ nhất để bắt đầu transcribe (1 giây audio)
MIN_AUDIO_SAMPLES = SAMPLE_RATE * 1


async def stream_transcribe(
    websocket: WebSocket,
    engine: WhisperEngine,
) -> None:
    """
    Nhận audio chunk qua WebSocket, gộp buffer rồi transcribe.
    Client gửi: bytes PCM float32 16kHz mono
    Server trả: JSON text result từng đoạn
    """
    await websocket.accept()
    logger.info("WebSocket client kết nối.")

    buffer = np.array([], dtype=np.float32)

    try:
        while True:
            data = await websocket.receive_bytes()

            # Nếu client gửi b"END" → flush buffer còn lại
            if data == b"END":
                if len(buffer) >= MIN_AUDIO_SAMPLES:
                    result = engine.transcribe(buffer)
                    await websocket.send_json({"event": "final", **result})
                await websocket.send_json({"event": "done"})
                break

            # Gộp chunk vào buffer
            chunk = np.frombuffer(data, dtype=np.float32)
            buffer = np.concatenate([buffer, chunk])

            # Đủ 5 giây thì transcribe một lần
            if len(buffer) >= SAMPLE_RATE * 5:
                result = engine.transcribe(buffer)
                await websocket.send_json({"event": "partial", **result})
                buffer = np.array([], dtype=np.float32)  # reset

    except WebSocketDisconnect:
        logger.info("WebSocket client ngắt kết nối.")
    except Exception as e:
        logger.error(f"Lỗi WebSocket: {e}")
        await websocket.send_json({"event": "error", "message": str(e)})
