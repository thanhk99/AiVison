import logging
import time
import threading
import os
import sys
import numpy as np
import pyaudio
import torch
from collections import deque

from core.config_loader import load_config
from engines.vision.vision_engine import VisionEngine
from engines.vision.security_engine import SecurityEngine
from engines.voice_engine import VoiceEngine
from engines.stt.audio_processor import SAMPLE_RATE

# Config logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MainAssistant")

# Audio Constants
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
VAD_THRESHOLD = 0.5
SILENCE_DURATION = 1.0

class MainAssistant:
    def __init__(self):
        self.config = load_config()
        
        # 1. Initialize Engines
        logger.info("[+] Khoi tao module Security (Face ID)...")
        self.security = SecurityEngine()
        
        logger.info("[+] Khoi tao module Voice (STT/LLM/TTS)...")
        self.voice = VoiceEngine(self.config)
        
        logger.info("[+] Khoi tao module Vision (Gesture)...")
        self.vision = VisionEngine(config=self.config.get("vision", {}))
        
        # 2. VAD Model for Voice
        logger.info("[+] Tai mo hinh VAD...")
        self.model_vad, self.vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                      model='silero_vad',
                                                      force_reload=False,
                                                      trust_repo=True)

        # 3. State Management
        self.is_unlocked = False
        self.current_user = "Unknown"
        self.is_running = False
        self.is_authenticating = False
        
        # Audio Stream
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start(self):
        self.is_running = True
        
        # Start Vision Engine
        self.vision.start()
        
        # Start Audio Loop
        self._audio_thread = threading.Thread(target=self._run_audio_loop, daemon=True)
        self._audio_thread.start()
        
        logger.info("[V] Quan gia Thanh da san sang. Hay ra hieu 'CALL' de bat dau.")
        
        try:
            while self.is_running:
                # Main logic loop for gesture and security coordination
                current_gesture = self.vision.get_current_gesture()
                
                if current_gesture == "CALL (Bat_dau)":
                    if not self.is_unlocked and not self.is_authenticating:
                        self._handle_authentication()
                
                elif current_gesture == "BAN_TAY (Huy)":
                    if self.is_unlocked:
                        self._lock_system()
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def _handle_authentication(self):
        """Thực hiện xác thực khuôn mặt khi có cử chỉ CALL."""
        self.is_authenticating = True
        logger.info(">>> Dang tien hanh xac thuc khuon mat...")
        
        # Lấy frame từ vision engine (camera đang mở)
        if self.vision.cap:
            ret, frame = self.vision.cap.read()
            if ret:
                user = self.security.authenticate(frame)
                if user != "Unknown" and user != "Guest":
                    self.is_unlocked = True
                    self.current_user = user
                    info = self.security.get_user_info()
                    msg = f"Chào {user}, Quản gia Thành đã sẵn sàng phục vụ. Bạn đang cảm thấy {info['emotion']} phải không?"
                    logger.info(f"[SEC] {msg}")
                    # Bỏ phần chào của hệ thống bằng giọng nói, chỉ hiển thị thôi
                else:
                    self.voice.speak("Xin lỗi, tôi không nhận diện được bạn. Vui lòng thử lại hoặc đăng ký khuôn mặt.")
            else:
                logger.error("Khong lay duoc frame tu camera de xac thuc.")
        
        self.is_authenticating = False

    def _lock_system(self):
        """Khóa hệ thống."""
        self.is_unlocked = False
        self.current_user = "Unknown"
        self.security.reset()
        logger.info("<<< He thong da KHOA.")
        self.voice.speak("Tạm biệt bạn, tôi sẽ quay lại trạng thái chờ.")

    def _run_audio_loop(self):
        """Vòng lặp xử lý âm thanh thời gian thực."""
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        audio_buffer = []
        silence_chunks = 0
        is_speaking = False
        pre_roll_chunks = int(SAMPLE_RATE / CHUNK * 0.5)
        pre_audio = deque(maxlen=pre_roll_chunks)
        max_silence_chunks = int(SAMPLE_RATE / CHUNK * SILENCE_DURATION)

        while self.is_running:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            
            # CHI XU LY GIONG NOI KHI DA MO KHOA VA AI KHONG DANG NOI
            if not self.is_unlocked or self.voice.tts.is_speaking:
                pre_audio.append(data)
                continue

            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            tensor_chunk = torch.from_numpy(audio_float32)
            
            with torch.no_grad():
                speech_prob = self.model_vad(tensor_chunk, SAMPLE_RATE).item()

            if speech_prob > VAD_THRESHOLD:
                if not is_speaking:
                    audio_buffer.extend(list(pre_audio))
                is_speaking = True
                silence_chunks = 0
                audio_buffer.append(data)
            elif is_speaking:
                silence_chunks += 1
                audio_buffer.append(data)

                if silence_chunks > max_silence_chunks:
                    # Chốt câu và xử lý
                    full_audio = b"".join(audio_buffer)
                    audio_np = np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Gọi Voice Engine xử lý quy trình STT -> LLM -> TTS
                    self.voice.process_voice_command(audio_np)
                    
                    is_speaking = False
                    audio_buffer = []
                    silence_chunks = 0
                    pre_audio.clear()
            else:
                pre_audio.append(data)

    def stop(self):
        self.is_running = False
        if self.vision:
            self.vision.stop()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        logger.info("He thong da dung.")

if __name__ == "__main__":
    assistant = MainAssistant()
    assistant.start()
