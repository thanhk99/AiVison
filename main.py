import logging
import time
import threading
import os
import sys
import numpy as np
import pyaudio
import torch
from collections import deque

from collections import deque
import tkinter as tk
from tkinter import simpledialog, messagebox
from core.config_loader import load_config
from core.mqtt_manager import MqttManager
from server.main import app as fastapi_app
import server.main as server_module
from engines.vision.vision_engine import VisionEngine
from engines.vision.security_engine import SecurityEngine
from engines.voice_engine import VoiceEngine
from engines.stt.audio_processor import SAMPLE_RATE
import cv2

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
        
        # 4. MQTT Manager
        self.mqtt = MqttManager(self.config)
        
        # Audio Stream
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.root = None # Tkinter root

    def show_menu(self):
        """Hiển thị menu chính bằng Tkinter GUI."""
        self.root = tk.Tk()
        self.root.title("QUẢN GIA THÀNH AI - PANEL")
        self.root.geometry("450x300")
        self.root.configure(bg="#f0f4f8")
        
        # Center window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')
        
        # UI Elements
        title_lbl = tk.Label(self.root, text="HỆ THỐNG QUẢN GIA THÀNH", font=("Segoe UI", 16, "bold"), bg="#f0f4f8", fg="#1e293b")
        title_lbl.pack(pady=20)
        
        btn_frame = tk.Frame(self.root, bg="#f0f4f8")
        btn_frame.pack(pady=10)
        
        btn_register = tk.Button(btn_frame, text="📸 Đăng ký khuôn mặt mới", font=("Segoe UI", 12, "bold"), 
                                 bg="#10b981", fg="white", activebackground="#059669", activeforeground="white",
                                 relief="flat", cursor="hand2", command=self._gui_run_face_registration, width=28, height=2)
        btn_register.pack(pady=10)
        
        btn_start = tk.Button(btn_frame, text="🔒 Khởi động Quản gia (Xác thực)", font=("Segoe UI", 12, "bold"), 
                              bg="#3b82f6", fg="white", activebackground="#2563eb", activeforeground="white",
                              relief="flat", cursor="hand2", command=self._gui_start_assistant, width=28, height=2)
        btn_start.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()

    def _gui_run_face_registration(self):
        name = simpledialog.askstring("Đăng Ký", "Nhập tên người dùng mới:", parent=self.root)
        if name and name.strip():
            self.root.withdraw()
            self._run_face_registration(name.strip())
            self.root.deiconify()
        elif name is not None:
            messagebox.showwarning("Cảnh báo", "Tên không hợp lệ.")

    def _gui_start_assistant(self):
        self.root.withdraw()
        if self._handle_initial_authentication():
            # Bắt đầu luồng kiểm soát chính
            threading.Thread(target=self.run_main_assistant, daemon=True).start()
        else:
            messagebox.showerror("Lỗi Xác Thực", "Không thể xác thực. Vui lòng thử lại!")
            self.root.deiconify()

    def _on_closing(self):
        if messagebox.askokcancel("Thoát", "Bạn có chắc chắn muốn dừng hệ thống hoàn toàn?", parent=self.root):
            self.stop()
            self.root.destroy()
            sys.exit(0)

    def _run_face_registration(self, name):
        """Luồng đăng ký khuôn mặt tích hợp."""
        print(f"\n--- ĐĂNG KÍ KHUÔN MẶT MỚI: {name} ---")

        # Sử dụng VisionEngine để mở camera
        self.vision.show_window = False
        self.vision.start()
        print(f"Đang chuẩn bị camera cho {name}...")
        print("Hướng dẫn: Nhấn 'S' trên cửa sổ camera để CHỤP, 'Q' để HỦY.")

        success = False
        while True:
            frame = self.vision.get_frame()
            if frame is not None:
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Dang ky: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Nhan 'S' de chup, 'Q' de huy", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Registration", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') or key == ord('S'):
                    print("Đang xử lý...")
                    from engines.vision.face_manager import FaceManager
                    face_mgr = FaceManager()
                    ok, msg = face_mgr.register_face(name, frame=frame)
                    if ok:
                        print(f"Thành công: {msg}")
                        success = True
                        break
                    else:
                        print(f"Thất bại: {msg}")
                elif key == ord('q') or key == ord('Q'):
                    print("Đã hủy.")
                    break
            time.sleep(0.01)
        
        self.vision.stop()
        cv2.destroyAllWindows()
        if success: time.sleep(1)

    def _handle_initial_authentication(self):
        """Xác thực khuôn mặt bắt buộc trước khi vào chế độ Quản gia."""
        print("\n>>> Vui lòng xác thực khuôn mặt để tiếp tục...")
        self.vision.show_window = False
        self.vision.start()
        
        attempts = 0
        max_attempts = 3
        authenticated = False

        while attempts < max_attempts and not authenticated:
            attempts += 1
            print(f"--- Lần thử {attempts}/{max_attempts} ---")
            
            # Đợi nhận diện (tối đa 30 frame)
            user = "Unknown"
            for _ in range(30):
                frame = self.vision.get_frame()
                if frame is not None:
                    cv2.imshow("Authentication", frame)
                    cv2.waitKey(1)
                    user = self.security.authenticate(frame)
                    if user != "Unknown" and user != "Guest":
                        authenticated = True
                        self.current_user = user
                        self.is_unlocked = True
                        break
                time.sleep(0.1)

            if not authenticated:
                print("Không nhận diện được khuôn mặt.")
                if attempts < max_attempts:
                    print("Vui lòng thử lại...")
                    time.sleep(1)

        cv2.destroyAllWindows()
        if authenticated:
            self.vision.show_window = True # Hiện lại window dành cho Vision
            print(f"Xác thực thành công! Chào mừng {self.current_user}.")
            msg = f"Chào mừng {self.current_user}, Quản gia Thành đã sẵn sàng."
            self.voice.speak(msg)
            return True
        else:
            print("Xác thực thất bại quá nhiều lần.")
            self.vision.stop()
            return False

    def run_main_assistant(self):
        """Khởi chạy luồng chính sau khi đã xác thực."""
        self.is_running = True
        
        # 1. Start Audio Loop
        self._audio_thread = threading.Thread(target=self._run_audio_loop, daemon=True)
        self._audio_thread.start()
        
        # 2. Start MQTT
        self.mqtt.set_command_callback(self._on_mqtt_command)
        self.mqtt.connect()
        
        # 3. Start FastAPI Server
        self._start_api_server()
        
        # Luồng chính là vòng lặp Vision (nhận diện cử chỉ tay)
        # VisionEngine đã được start trong _handle_initial_authentication
        
        logger.info("[V] Hệ thống Quản gia đang hoạt động. Nhấn 'L' hoặc 'Q' trên cửa sổ camera để khóa.")
        
        try:
            while self.is_running:
                # 1. Lấy phím bấm từ Vision Engine
                key = self.vision.get_last_key()
                
                if key in (ord('l'), ord('L'), ord('q'), ord('Q')):
                    if self.is_unlocked:
                        self._lock_system()
                        # Khi khóa quay lại menu
                        if self.root:
                            self.root.after(0, self.root.deiconify)
                        break
                
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.stop()

    def _lock_system(self):
        """Khóa hệ thống và dừng các tiến trình con để quay lại menu."""
        self.is_running = False
        self.is_unlocked = False
        self.current_user = "Unknown"
        self.security.reset()
        
        # Dừng các engine
        self.vision.stop()
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except: pass
            self.stream = None
        self.mqtt.stop()
        
        logger.info("<<< Hệ thống đã KHÓA và quay lại menu chờ.")
        self.voice.speak("Tạm biệt bạn, tôi đã khóa hệ thống và quay lại trạng thái chờ.")
        cv2.destroyAllWindows()

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
        audio_buffer = []
        silence_chunks = 0
        is_speaking = False
        pre_roll_chunks = int(SAMPLE_RATE / CHUNK * 0.5)
        pre_audio = deque(maxlen=pre_roll_chunks)
        max_silence_chunks = int(SAMPLE_RATE / CHUNK * SILENCE_DURATION)

        while self.is_running:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            
            # CHI XU LY GIONG NOI KHI DA MO KHOA
            if not self.is_unlocked:
                pre_audio.clear()
                audio_buffer = []
                is_speaking = False
                silence_chunks = 0
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
        self.vision.stop()
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except: pass
        self.p.terminate()
        self.mqtt.stop()
        cv2.destroyAllWindows()
        logger.info("Hệ thống đã dừng hoàn toàn.")

    def _on_mqtt_command(self, data):
        """Xử lý lệnh từ MQTT."""
        if isinstance(data, str):
            command = data.lower()
        elif isinstance(data, dict):
            command = data.get("command", "").lower()
        else:
            return

        logger.info(f"[MQTT] Nhan lenh: {command}")
        
        if command == "unlock":
            if not self.is_unlocked:
                # Mo khoa gia lap hoac kich hoat camera xac thuc
                logger.info("[MQTT] Kich hoat xac thuc tu xa...")
                # self._handle_authentication() # Can than vi chay trong thread khac
        elif command == "lock":
            if self.is_unlocked:
                self._lock_system()
        elif "bật đèn" in command:
             # Vi du ve event
             self.mqtt.publish_event("iot_control", {"device": "light", "action": "on"})
             self.voice.speak("Đã bật đèn cho bạn.")
        elif "tắt đèn" in command:
             self.mqtt.publish_event("iot_control", {"device": "light", "action": "off"})
             self.voice.speak("Đã tắt đèn cho bạn.")

    def _start_api_server(self):
        """Khởi chạy FastAPI server trong thread riêng."""
        import uvicorn
        
        # Gán instance assistant cho server module để các endpoint có thể truy cập
        server_module.assistant = self
        
        host = self.config.get("server", {}).get("host", "0.0.0.0")
        port = self.config.get("server", {}).get("port", 8000)
        
        def run_server():
            logger.info(f"[+] Khoi tao API Server tai http://{host}:{port}")
            uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")
            
        self._api_thread = threading.Thread(target=run_server, daemon=True)
        self._api_thread.start()

if __name__ == "__main__":
    assistant = MainAssistant()
    assistant.show_menu()
