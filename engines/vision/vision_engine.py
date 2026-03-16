import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
import threading
import time
import sys
import os

abs_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.config_loader import load_config

logger = logging.getLogger("VisionEngine")

class VisionEngine:
    def __init__(self, config=None):
        """
        Khởi tạo Vision Engine sử dụng MediaPipe Hands.
        """
        cfg = config or load_config().get("vision", {})
        self.camera_id = cfg.get("camera_id", 0)
        self.min_detection_confidence = cfg.get("min_detection_confidence", 0.7)
        self.min_tracking_confidence = cfg.get("min_tracking_confidence", 0.5)
        self.show_window = cfg.get("show_window", False)
        
        # Khởi tạo MediaPipe Tasks
        model_path = os.path.join(project_root, "models", "vision", "hand_landmarker.task")
        if not os.path.exists(model_path):
            logger.error(f"Khong tim thay model: {model_path}")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.mp_draw = None # MediaPipe Tasks khong di kem drawing utils don gian nhu solutions
        
        self.current_gesture = None
        self.cap = None
        self.running = False
        self.thread = None

        # Trạng thái máy (State Machine)
        self.is_active = False # Bật khi thấy CALL
        self.pending_gesture = "None" # Lưu lệnh đang chờ thực thi

        self.gesture_history = [] 
        self.required_frames = 6 
        self.lost_hand_count = 0 
        self.max_lost_frames = 15 
        self.last_key = -1  # Lưu phím bấm cuối cùng

    def start(self):
        """Bắt đầu luồng nhận diện camera."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open_camera(self, cam_id):
        """Hàm helper để mở camera với nhiều backend khác nhau."""
        if cam_id is None: return None
        
        
        if isinstance(cam_id, int):
            backends = [cv2.CAP_DSHOW, None, cv2.CAP_MSMF]
        else:
            backends = [None]
        
        for backend in backends:
            name = "Default" if backend is None else ("DSHOW" if backend == cv2.CAP_DSHOW else "MSMF")
            try:
                logger.info(f"Dang thu mo Camera {cam_id} voi backend: {name}...")
                if backend is not None:
                    cap = cv2.VideoCapture(cam_id + backend)
                else:
                    cap = cv2.VideoCapture(cam_id)
                
                if cap.isOpened():
                    # Kiem tra xem co doc duoc frame khong (tranh man hinh xanh)
                    ret, frame = cap.read()
                    if ret:
                        logger.info(f"[V] Da mo thanh cong camera {cam_id} bang {name}")
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        return cap
                    else:
                        logger.warning(f"Backend {name} mo duoc nhung khong doc duoc hinh anh.")
                        cap.release()
            except Exception as e:
                logger.error(f"Loi khi thu backend {name}: {e}")
                
        return None

    def stop(self):
        """Dừng nhận diện."""
        self.running = False
        time.sleep(0.5) # Chờ thread dừng
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.show_window:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        logger.info("Vision Engine da dung.")

    def _run(self):
        self.cap = self._open_camera(self.camera_id)
        
        if not self.cap or not self.cap.isOpened():
            logger.error("Khong the mo camera ban dau.")
            self.running = False
            return

        logger.info("Vision Engine da khoi chay thanh cong.")

        while self.running:
            if not self.cap: break
            success, img = self.cap.read()
            
            if not success:
                logger.warning("Khong thể đọc dữ liệu từ camera. Dang thu lai...")
                time.sleep(1)
                continue

            # Chuyển sang RGB cho MediaPipe và tạo MediaPipe Image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            # Nhận diện
            detection_result = self.detector.detect(mp_image)

            raw_gesture = "None"
            stable_gesture = "None"

            if detection_result.hand_landmarks:
                self.lost_hand_count = 0 
                for hand_landmarks in detection_result.hand_landmarks:
                    if self.show_window:
                        # Vẽ landmarks thủ công (vì không còn drawing_utils mượt như trước)
                        # Chúng ta vẽ các khớp chính
                        for lm in hand_landmarks:
                            x_px, y_px = int(lm.x * 640), int(lm.y * 480)
                            cv2.circle(img, (x_px, y_px), 3, (0, 255, 0), -1)
                    
                    current_raw = self._get_gesture(hand_landmarks)
                    
                    self.gesture_history.append(current_raw)
                    if len(self.gesture_history) > self.required_frames:
                        self.gesture_history.pop(0)
                    
                    if len(self.gesture_history) == self.required_frames and len(set(self.gesture_history)) == 1:
                        stable_gesture = self.gesture_history[0]
            else:
                # Mất dấu tay: Chỉ xóa lịch sử cử chỉ, KHÔNG tự động hủy trạng thái ACTIVE
                self.gesture_history = []

            # --- QUY TRÌNH NHẬN DIỆN CỬ CHỈ ---
            self.current_gesture = stable_gesture if stable_gesture != "None" else "Waiting..."

            if self.show_window:
                cv2.rectangle(img, (0, 0), (640, 45), (30, 30, 30), -1)
                cv2.putText(img, f"GESTURE: {self.current_gesture}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(img, "CALL: Bat dau | BAN_TAY: Huy | NAM_TAY: Thuc thi", (10, 465), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("AI Home Assistant Vision", img)
                
                key = cv2.waitKey(1) & 0xFF
                self.last_key = key # Lưu lại để MainAssistant truy cập
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('n') and isinstance(self.camera_id, int):
                    self.cap.release()
                    self.camera_id = (self.camera_id + 1) % 3
                    self.cap = self._open_camera(self.camera_id)

        self.stop()

    def _get_gesture(self, hand_landmarks):
        """
        Nhận diện cử chỉ máy trạng thái: CALL, BAN_TAY, NAM_TAY và các lệnh điều hướng.
        hand_landmarks là một list các đối tượng có thuộc tính x, y, z.
        """
        lm = hand_landmarks
        
        # Kiểm tra ngón cái (Thumb)
        thumb_open = lm[4].y < lm[3].y and abs(lm[4].x - lm[2].x) > 0.05
        
        # 4 ngón còn lại
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        fingers = [1 if thumb_open else 0]
        for i in range(4):
            if lm[tips[i]].y < lm[pips[i]].y:
                fingers.append(1)
            else:
                fingers.append(0)

        total_open = sum(fingers)
        idx, mid, ring, pnk = fingers[1], fingers[2], fingers[3], fingers[4]

        # --- LOGIC NHẬN DIỆN MỚI ---
        
        # 1. BAN_TAY (Mở 5 ngón) -> Đã đổi thành HỦY
        if total_open >= 4:
            return "BAN_TAY (Huy)"
            
        # 2. NAM_TAY (Khép hết) -> Đã đổi thành THỰC THI
        if total_open == 0:
            return "NAM_TAY (Thuc_thi)"

        # 3. CALL (Ngón cái + Ngón út) -> Kích hoạt
        if thumb_open and pnk and not idx and not mid and not ring:
            return "CALL (Bat_dau)"

        # 4. CÁC LỆNH ĐIỀU HƯỚNG (Khi đang lắng nghe)
        if idx and total_open == 1:
            return "MOT_NGON (Ve_truoc)"
            
        if idx and mid and total_open == 2:
            return "HAI_NGON (Tiep_theo)"
            
        if mid and total_open == 1:
            return "NGON_GIUA (Im_lang)"

        return "Unknown"

    def get_current_gesture(self):
        return self.current_gesture

    def get_last_key(self):
        """Lấy phím bấm gần nhất và reset nó."""
        key = self.last_key
        self.last_key = -1
        return key

if __name__ == "__main__":
    # Cấu hình logging để thấy output trong console
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Đọc cấu hình thật từ config.yaml
    real_config = load_config().get("vision", {})
    # Đảm bảo bật cửa sổ để test
    real_config["show_window"] = True
    
    vision = VisionEngine(config=real_config)
    try:
        vision.start()
        
        # Cho mot chut de camera khoi tao xong trong thread rieng
        print("Dang doi camera san sang...", end="")
        timeout = 10 
        start_wait = time.time()
        while not vision.cap and (time.time() - start_wait < timeout):
            print(".", end="", flush=True)
            time.sleep(0.5)
            
        if vision.cap and vision.cap.isOpened():
            print("\n" + "="*40)
            print("DANG CHAY RIENG MODULE DIEU KHIEN TAY")
            print("Nhan 'q' trong cua so camera hoac Ctrl+C de dung.")
            print("="*40 + "\n")
            
            while vision.running:
                gesture = vision.get_current_gesture()
                if gesture and gesture != "None":
                    print(f"Phat hien cu chi: {gesture}      ", end="\r")
                time.sleep(0.1)
        else:
            print("\n\n[!] Khong the khoi dong camera sau 10 giay.")
            print("Goi y: Hay chac chan DroidCam PC Client da nhan 'Start'.")
            
    except KeyboardInterrupt:
        vision.stop()
    finally:
        print("\nDa dung module Vision.")
