import logging
import time
import os
import yaml
from collections import deque

import numpy as np
import pyaudio
import torch

# Tắt cảnh báo symlink của HuggingFace
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from core.config_loader import load_config
from engines.stt.whisper_engine import WhisperEngine
from engines.stt.audio_processor import SAMPLE_RATE
from engines.llm.llm_engine import LLMEngine
from engines.tts.tts_engine import TTSEngine
from engines.vision.vision_engine import VisionEngine

# ── Cấu hình Audio ──
CHUNK = 512  # Cỡ chunk mượt cho VAD
FORMAT = pyaudio.paInt16
CHANNELS = 1

# VAD confidence (Độ tự tin phát hiện giọng nói của AI VAD - từ 0 đến 1)
VAD_THRESHOLD = 0.5  

# Thời gian im lặng tối đa (giây) trước khi chốt câu và gửi đi nhận diện
SILENCE_DURATION = 1.0   

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("RealtimeSTT")


def start_realtime_transcription():
    # Load config thông qua core loader
    config = load_config()
    
    # 1. Khởi tạo model Whisper (chạy offline trên máy)
    logger.info("[+] Dang tai mo hinh nhan dien STT AI...")
    engine = WhisperEngine()
    
    # 2. Khởi tạo mô hình VAD lọc tiếng ồn (Silero VAD)
    logger.info("[+] Dang tai mo hinh loc am VAD...")
    model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      trust_repo=True)
    
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    # 3. Khởi tạo LLM Engine (Ollama)
    llm_conf = config.get("assistant", {})
    llm = LLMEngine(
        base_url=llm_conf.get("ollama_url", "http://localhost:11434"),
        model=llm_conf.get("model_size", "gemma2:2b")
    )
    llm.system_prompt = llm_conf.get("system_prompt", llm.system_prompt)
    
    # 4. Khởi tạo TTS Engine (Phát âm thanh cao cấp Piper)
    tts_conf = config.get("tts", {})
    tts = TTSEngine(
        model_path=tts_conf.get("model_path"),
        config_path=tts_conf.get("config_path")
    )

    # 5. Khởi tạo Vision Engine (Nhìn cử chỉ tay)
    vision_conf = config.get("vision", {})
    vision = None
    if vision_conf.get("enabled", True):
        logger.info("[+] Dang tai mo hinh thi giac Vision...")
        vision = VisionEngine(config=vision_conf)
        vision.start()
        
    logger.info("[V] Mo hinh da san sang. Hay bat dau noi!")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    audio_buffer = []       # Chứa các chunk audio trong một câu
    silence_chunks = 0      # Đếm số chunk im lặng liên tiếp
    is_speaking = False     # Trạng thái có đang nói hay không
    
    # Số lượng chunk tương đương 0.5s âm thanh "chuẩn bị" (pre-roll)
    pre_roll_chunks = int(SAMPLE_RATE / CHUNK * 0.5)
    pre_audio = deque(maxlen=pre_roll_chunks) # Hàng đợi lưu âm thanh trước khi nói
    
    max_silence_chunks = int(SAMPLE_RATE / CHUNK * SILENCE_DURATION)

    print("\n" + "="*50)
    print("[Mic] Dang nghe... (Nhan Ctrl+C de thoat)")
    print("="*50 + "\n")

    try:
        while True:
            # Đọc audio từ mic
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert bytes sang Float Tensor để đưa vào mô hình VAD
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            tensor_chunk = torch.from_numpy(audio_float32)
            
            # Phân tích bằng VAD Model để xem có phải là giọng người không
            with torch.no_grad():
                speech_prob = model_vad(tensor_chunk, SAMPLE_RATE).item()

            # --- Kiểm tra cử chỉ tay (Vision) ---
            if vision:
                gesture = vision.get_current_gesture()
                if gesture == "Open Hand":
                    tts.stop() # Dừng nói ngay khi giơ tay
                
            if speech_prob > VAD_THRESHOLD:
                # Đang có người kể cả nói thầm
                if not is_speaking:
                    # Vừa mới bắt đầu nói -> Nạp toàn bộ âm thanh đệm 0.5s vào trước
                    audio_buffer.extend(list(pre_audio))
                
                is_speaking = True
                silence_chunks = 0
                audio_buffer.append(data)
                print(".", end="", flush=True)  # Hiển thị dấu chấm báo hiệu đang thu âm

            elif is_speaking:
                # Nếu đã bắt đầu nói nhưng VAD phát hiện hiện tại đang im lặng (hoặc chỉ toàn tiếng ồn/tiếng quạt)
                silence_chunks += 1
                audio_buffer.append(data)

                # Nếu im lặng khóa quá thời gian cho phép -> chốt câu
                if silence_chunks > max_silence_chunks:
                    print("\n[+] Dang xu ly...", end="\r")
                    
                    # Gộp tất cả data đã thu được
                    full_audio = b"".join(audio_buffer)
                    
                    # Convert byte PCM Int16 sang Numpy Float32 cho Whisper
                    audio_np = np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Lọc bằng Whisper STT
                    start_time = time.time()
                    result = engine.transcribe(audio_np)
                    processing_time = time.time() - start_time
                    
                    text = result.get('text', '').strip()
                    
                    # Xoá dòng "Đang xử lý..."
                    print(" "*50, end="\r") 
                    
                    if text:
                        print(f"[YOU] {text}  ({processing_time:.2f}s)")
                        
                        # --- Gửi qua LLM để lấy phản hồi ---
                        print(f"[AI] ", end="", flush=True)
                        ai_response = ""
                        for chunk in llm.generate_response_stream(text):
                            print(chunk, end="", flush=True)
                            ai_response += chunk
                        print("\n")
                        
                        # --- Phát âm thanh trả lời ---
                        if ai_response:
                            tts.speak(ai_response)
                    
                    # Reset để chuẩn bị nghe câu mới
                    is_speaking = False
                    audio_buffer = []
                    silence_chunks = 0
                    pre_audio.clear()
            else:
                # Đang im lặng hoàn toàn, liên tục lưu vào bộ đệm 0.5s quay vòng
                pre_audio.append(data)

    except KeyboardInterrupt:
        print("\n[-] Da dung nhan dien.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    start_realtime_transcription()
