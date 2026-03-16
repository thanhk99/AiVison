import logging
import time
from typing import Generator
import numpy as np
from engines.stt.whisper_engine import WhisperEngine
from engines.llm.llm_engine import LLMEngine
from engines.tts.tts_engine import TTSEngine

logger = logging.getLogger("VoiceEngine")

class VoiceEngine:
    def __init__(self, config):
        """
        Khoi tao Voice Engine ket hop STT, LLM va TTS.
        """
        logger.info("Dang khoi tao Voice Engine...")
        
        # 1. Whisper STT
        self.stt = WhisperEngine(config=config.get("whisper", {}))
        
        # 2. Ollama LLM
        llm_conf = config.get("assistant", {})
        self.llm = LLMEngine(
            base_url=llm_conf.get("ollama_url", "http://localhost:11434"),
            model=llm_conf.get("model_size", "gemma2:2b")
        )
        self.llm.system_prompt = llm_conf.get("system_prompt", "")
        
        # 3. Piper TTS
        tts_conf = config.get("tts", {})
        self.tts = TTSEngine(
            model_path=tts_conf.get("model_path"),
            config_path=tts_conf.get("config_path")
        )
        
        logger.info("Voice Engine da san sang.")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Chuyen doi am thanh sang van ban."""
        result = self.stt.transcribe(audio_data)
        return result.get("text", "").strip()

    def generate_response(self, text: str) -> Generator[str, None, None]:
        """Gui van ban den LLM va nhan phan hoi stream."""
        return self.llm.generate_response_stream(text)

    def speak(self, text: str, async_mode: bool = True):
        """Phat am thanh tu van ban."""
        if async_mode:
            self.tts.speak_async(text)
        else:
            self.tts.speak(text)

    def stop_speaking(self):
        """Dung phát thanh ngay lap tuc."""
        self.tts.stop()

    def process_voice_command(self, audio_data: np.ndarray):
        """
        Quy trinh day chuyen: STT -> LLM -> TTS.
        Ham nay thuc hien dong bo de de quan ly trong vong lap main.
        """
        text = self.transcribe(audio_data)
        if not text:
            return None
            
        logger.info(f"[USER] {text}")
        
        print(f"[AI] ", end="", flush=True)
        full_response = ""
        for chunk in self.generate_response(text):
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n")
        
        if full_response:
            self.speak(full_response)
            
        return full_response
