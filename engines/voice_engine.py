import logging
import json
import numpy as np
import sounddevice as sd
from typing import Optional
from supertonic import TTS

from engines.stt.whisper_engine import WhisperEngine
from engines.llm.llm_engine import LLMEngine

logger = logging.getLogger("VoiceEngine")


class VoiceEngine:
    def __init__(self, config):
        """Khởi tạo Voice Engine kết hợp STT, LLM và TTS."""
        logger.info("Dang khoi tao Voice Engine...")

        # STT – Whisper
        self.stt = WhisperEngine(config=config.get("whisper", {}))

        # LLM – Ollama
        llm_conf = config.get("assistant", {})
        self.llm = LLMEngine(
            base_url=llm_conf.get("llm_base_url", "http://localhost:6011"),
            model=llm_conf.get("model_size", "deepseek-v4-flash-nothinking")
        )
        self.llm.system_prompt = llm_conf.get("system_prompt", "")

        # Từ khoá wake word và end word (đọc từ config)
        self.wake_words = [w.lower() for w in llm_conf.get("wake_words", ["thành ơi", "quản gia"])]
        self.end_words  = [w.lower() for w in llm_conf.get("end_words",  ["tạm biệt", "xong rồi"])]

        # TTS – Supertonic
        tts_conf = config.get("supertonic", {})
        self.tts_enabled = tts_conf.get("enabled", False)
        self.is_speaking = False

        if self.tts_enabled:
            logger.info("Dang tai mo hinh Supertonic TTS...")
            self.tts = TTS(auto_download=True)
            voice_style_name = tts_conf.get("voice_style", "M1")
            self.voice_style = self.tts.get_voice_style(voice_style_name)
            self.tts_speed = tts_conf.get("speed", 1.05)
            self.tts_steps = tts_conf.get("total_steps", 8)
            logger.info(f"Supertonic TTS da san sang (Voice: {voice_style_name}).")
        else:
            self.tts = None
            logger.info("Voice Engine da san sang (Khong su dung TTS).")

    # ------------------------------------------------------------------ #
    #  Transcribe                                                          #
    # ------------------------------------------------------------------ #
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Chuyển đổi âm thanh sang văn bản."""
        result = self.stt.transcribe(audio_data)
        return result.get("text", "").strip()

    # ------------------------------------------------------------------ #
    #  Wake Word / End Word Detection                                      #
    # ------------------------------------------------------------------ #
    def check_wake_word(self, text: str) -> bool:
        """Kiểm tra xem text có chứa wake word không."""
        text_lower = text.lower()
        for word in self.wake_words:
            if word in text_lower:
                logger.info(f"[WAKE] Phát hiện wake word: '{word}'")
                return True
        return False

    def check_end_word(self, text: str) -> bool:
        """Kiểm tra xem text có chứa end word không (fallback nếu LLM không trả SESSION:END)."""
        text_lower = text.lower()
        for word in self.end_words:
            if word in text_lower:
                logger.info(f"[END] Phát hiện end word: '{word}'")
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Parse LLM Response                                                  #
    # ------------------------------------------------------------------ #
    def _parse_llm_response(self, raw: str) -> dict:
        """
        Parse chuỗi JSON từ LLM.
        Fallback an toàn nếu LLM không trả đúng định dạng.
        """
        try:
            data = json.loads(raw)
            message = data.get("message", "").strip()
            command = data.get("command", "NONE").strip()
            if not message:
                message = raw
            return {"message": message, "command": command}
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"[LLM] Không parse được JSON, dùng raw text: {raw[:80]}")
            # Nếu text chứa end word → tự động kết thúc phiên
            if self.check_end_word(raw):
                return {"message": raw, "command": "SESSION:END"}
            return {"message": raw, "command": "NONE"}

    # ------------------------------------------------------------------ #
    #  TTS                                                                 #
    # ------------------------------------------------------------------ #
    def speak(self, text: str, async_mode: bool = True):
        """Phát âm thanh từ văn bản sử dụng Supertonic."""
        if not self.tts_enabled or not self.tts:
            return

        logger.info(f"[TTS] Dang noi: '{text}'")
        
        try:
            wav, _ = self.tts.synthesize(
                text=text,
                lang="vi",
                voice_style=self.voice_style,
                total_steps=self.tts_steps,
                speed=self.tts_speed
            )
            
            def play_audio():
                self.is_speaking = True
                try:
                    sd.play(wav.squeeze(), samplerate=44100)
                    sd.wait()
                finally:
                    self.is_speaking = False

            if async_mode:
                import threading
                threading.Thread(target=play_audio, daemon=True).start()
            else:
                play_audio()
                
        except Exception as e:
            logger.error(f"Loi TTS: {e}")

    def stop_speaking(self):
        """Dừng phát thanh ngay lập tức."""
        sd.stop()
        self.is_speaking = False

    # ------------------------------------------------------------------ #
    #  Core Processing                                                     #
    # ------------------------------------------------------------------ #
    def _process(self, text: str) -> Optional[dict]:
        """
        Luồng chính: text → LLM → parse JSON → {message, command}
        """
        if not text:
            return None

        raw = self.llm.generate_response_full(text)
        result = self._parse_llm_response(raw)

        message = result["message"]
        command = result["command"]

        logger.info(f"[AI] message : {message}")
        logger.info(f"[AI] command : {command}")
        print(f"\n[AI] 💬 {message}")
        print(f"[AI] ⚙️  {command}\n")

        if message:
            self.speak(message)

        return result

    def process_voice_command(self, audio_data: np.ndarray) -> Optional[dict]:
        """
        Quy trình đầy đủ: Audio → STT → LLM → parse → TTS.
        Trả về dict {message, command}.
        """
        text = self.transcribe(audio_data)
        if not text:
            return None

        logger.info(f"[USER] {text}")
        return self._process(text)

    def process_text_command(self, text: str) -> Optional[dict]:
        """
        Xử lý lệnh từ văn bản (bỏ qua STT), dùng cho API/MQTT.
        Trả về dict {message, command}.
        """
        logger.info(f"[API/MQTT Input] {text}")
        return self._process(text)
