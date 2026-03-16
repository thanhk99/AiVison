import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# Fix lỗi symlink trên Windows khi cache HuggingFace model
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from faster_whisper import WhisperModel

from core.config_loader import load_config
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

def _load_config() -> dict:
    return load_config().get("whisper", {})


class WhisperEngine:
    """Wrapper Faster-Whisper cho nhận diện tiếng Việt."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or _load_config().get("whisper", {})

        model_size = cfg.get("model_size", "small")
        device = cfg.get("device", "cpu")
        compute_type = cfg.get("compute_type", "int8")

        logger.info(f"Đang tải model Whisper: {model_size} | device={device} | compute={compute_type}")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

        self.language = cfg.get("language", "vi")
        self.beam_size = cfg.get("beam_size", 5)
        self.vad_filter = cfg.get("vad_filter", True)

        logger.info("Model Whisper đã sẵn sàng.")

    def transcribe(self, audio: np.ndarray) -> dict:
        """
        Nhận diện giọng nói từ numpy array float32 16kHz.
        Trả về dict gồm text và các segment.
        """
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False # Thiết lập quan trọng cho Realtime STT để chống lỗi ảo giác
        )

        result_segments = []
        full_text_parts = []

        for seg in segments:
            result_segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

        return {
            "text": " ".join(full_text_parts),
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
            "segments": result_segments,
        }
