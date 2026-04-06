import os
import logging
import cv2
import time
import shutil
import unicodedata
from deepface import DeepFace
import sys

# Thêm project root vào sys.path để import core
abs_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.config_loader import load_config

logger = logging.getLogger("FaceManager")

class FaceManager:
    def __init__(self, faces_dir="models/vision/faces"):
        self.faces_dir = faces_dir
        
        # Tải cấu hình
        config = load_config().get("vision", {})
        self.model_name = config.get("face_model", "ArcFace")
        self.detector_backend = config.get("face_detector", "opencv")
        self.threshold = config.get("face_threshold", 0.35)
        self.anti_spoofing = config.get("anti_spoofing", True)
        
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)

    def remove_accents(self, input_str):
        """Xóa dấu tiếng Việt và chuẩn hóa chuỗi."""
        if not input_str:
            return ""
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def register_face(self, name, image_path=None, frame=None):
        """
        Đăng ký một khuôn mặt mới bằng cách lưu ảnh vào thư mục faces/.
        DeepFace sẽ quét thư mục này để nhận diện.
        """
        try:
            # Chuẩn hóa tên để tạo thư mục an toàn
            safe_name = self.remove_accents(name).strip().replace(" ", "_").lower()
            user_dir = os.path.join(self.faces_dir, safe_name)
            
            # --- KIỂM TRA TRÙNG LẶP ---
            try:
                check_input = frame if frame is not None else image_path
                if check_input is not None:
                    existing_user = self.identify_face(check_input)
                    if existing_user != "Unknown":
                        if existing_user.lower() != safe_name:
                            logger.warning(f"Tu choi dang ky: {name} trung khop voi {existing_user}")
                            return False, f"Khuôn mặt này đã được đăng ký dưới tên: {existing_user}"
            except Exception as e:
                logger.debug(f"Bo qua loi kiem tra trung lap: {e}")
            # ---------------------------

            if not os.path.exists(user_dir):
                os.makedirs(user_dir)
            
            # Tạo tên file duy nhất
            timestamp = int(time.time())
            filename = f"{safe_name}_{timestamp}.jpg"
            target_path = os.path.join(user_dir, filename)
            
            if image_path:
                shutil.copy(image_path, target_path)
            elif frame is not None:
                # Kiểm tra độ mờ (blur check)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                if variance < 80:
                    return False, "Ảnh quá mờ! Vui lòng giữ khuôn mặt tĩnh."
                
                # Lưu ảnh chất lượng cao
                is_success, buffer = cv2.imencode(".jpg", frame)
                if is_success:
                    with open(target_path, "wb") as f:
                        f.write(buffer)
                else:
                    return False, "Không thể mã hóa hình ảnh."
            else:
                return False, "Không có dữ liệu hình ảnh."

            # Kiểm tra khuôn mặt và Anti-spoofing
            try:
                objs = DeepFace.extract_faces(
                    img_path=target_path, 
                    detector_backend=self.detector_backend, 
                    enforce_detection=True, 
                    anti_spoofing=self.anti_spoofing
                )
                
                if self.anti_spoofing:
                    is_real = all(obj.get("is_real", True) for obj in objs)
                    if not is_real:
                        if os.path.exists(target_path): os.remove(target_path)
                        return False, "Phát hiện khuôn mặt giả mạo (Spoofing)!"
                
                if len(objs) > 0:
                    logger.info(f"Da dang ky thanh cong: {name} (Safe name: {safe_name})")
                    return True, f"Đã đăng ký {name} thành công."
                else:
                    if os.path.exists(target_path): os.remove(target_path)
                    return False, "Không tìm thấy khuôn mặt trong ảnh."
                    
            except Exception as e:
                if os.path.exists(target_path): os.remove(target_path)
                return False, f"Lỗi nhận diện khuôn mặt: {e}"
                
        except Exception as e:
            return False, f"Lỗi hệ thống: {e}"

    def identify_face(self, frame):
        """Nhận diện khuôn mặt trong một frame hình ảnh sử dụng DeepFace.find."""
        try:
            # Nhận diện
            results = DeepFace.find(
                img_path=frame, 
                db_path=self.faces_dir, 
                model_name=self.model_name, 
                detector_backend=self.detector_backend, 
                distance_metric='cosine',
                enforce_detection=False,
                silent=True
            )
            
            if len(results) > 0 and not results[0].empty:
                best_match = results[0].iloc[0]
                
                # Tìm cột distance phù hợp
                dist_col = None
                for col in results[0].columns:
                    if 'cosine' in col.lower() or 'distance' in col.lower():
                        dist_col = col
                        break
                
                if dist_col is None:
                    return "Unknown"
                
                distance = best_match[dist_col]
                
                if distance < self.threshold:
                    best_match_path = best_match['identity']
                    user_name = os.path.basename(os.path.dirname(best_match_path))
                    logger.info(f"Nhan dien: {user_name} (dist: {distance:.4f})")
                    return user_name
                
            return "Unknown"
        except Exception as e:
            # logger.error(f"Loi khi nhan dien: {e}")
            return "Unknown"
