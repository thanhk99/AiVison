import pyaudio
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def calculate_rms(data: bytes) -> float:
    audio_data = np.frombuffer(data, dtype=np.int16)
    if len(audio_data) == 0: return 0.0
    return np.sqrt(np.mean(np.square(audio_data.astype(np.float64))))

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Dang do am luong micro... (Nhan Ctrl+C de thoat)")
try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        rms = calculate_rms(data)
        # In ra biểu đồ độ lớn âm thanh bằng dấu |
        bar = "|" * int(rms / 2)  # Chia 2 để thanh không quá dài
        print(f"\rAm luong: {rms:6.1f} {bar:<50}", end="", flush=True)
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
