import os
import logging
import threading
import wave
import json
import numpy as np
import pyaudio
from piper.voice import PiperVoice

from core.config_loader import load_config
from piper.voice import PiperVoice

logger = logging.getLogger("TTSEngine")

class TTSEngine:
    def __init__(self, model_path=None, config_path=None):
        """
        Khởi tạo engine Piper TTS.
        """
        # Load config mặc định nếu không truyền vào
        cfg = load_config().get("tts", {})
        
        self.model_path = model_path or cfg.get("model_path") or "models/tts/vi_VN-vais1000-medium.onnx"
        self.config_path = config_path or cfg.get("config_path") or (self.model_path + ".json")

        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Khong tim thay model Piper tai: {self.model_path}")
            
            logger.info(f"Dang tai model Piper TTS: {self.model_path}")
            self.voice = PiperVoice.load(self.model_path, config_path=self.config_path)
            
            # Khởi tạo PyAudio
            self.p = pyaudio.PyAudio()
            self.is_speaking = False
        except Exception as e:
            logger.error(f"Loi khi khoi tao Piper TTS: {e}")
            self.voice = None
            self.is_speaking = False

    def speak(self, text):
        """
        Phát âm thanh từ văn bản bằng Piper.
        Piper sinh ra âm thanh PCM 16-bit mono.
        """
        if not self.voice or not text:
            return
            
        self.stop_flag = False
        self.is_speaking = True # Bat trang thai dang noi
        try:
            # Stream âm thanh từ Piper
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.voice.config.sample_rate,
                output=True
            )
            
            # Piper sinh audio và chúng ta đẩy trực tiếp vào stream của loa
            for chunk in self.voice.synthesize(text):
                if self.stop_flag:
                    logger.info("TTS dung theo yeu cau.")
                    break
                stream.write(chunk.audio_int16_bytes)
                
            stream.stop_stream()
            stream.close()
            self.stop_flag = False # Reset sau khi xong
            
        except Exception as e:
            logger.error(f"Lỗi khi phát âm thanh Piper: {e}")
        finally:
            self.is_speaking = False # Luon reset ve False khi ket thuc

    def stop(self):
        """Dừng phát âm thanh ngay lập tức."""
        self.stop_flag = True

    def speak_async(self, text):
        """
        Phát âm thanh trong một thread riêng.
        """
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()

    def __del__(self):
        if hasattr(self, 'p'):
            self.p.terminate()
