import os
import logging
import cv2
from deepface import DeepFace

logger = logging.getLogger("FaceManager")

class FaceManager:
    def __init__(self, faces_dir="models/vision/faces"):
        self.faces_dir = faces_dir
        self.model_name = "VGG-Face" # Model can bang giua toc do va do chinh xac
        
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)

    def register_face(self, name, image_path=None, frame=None):
        """
        Đăng ký một khuôn mặt mới bằng cách lưu ảnh vào thư mục faces/.
        DeepFace sẽ quét thư mục này để nhận diện.
        """
        try:
            user_dir = os.path.join(self.faces_dir, name)
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)
            
            # Lưu ảnh vào thư mục người dùng
            timestamp = int(os.time()) if hasattr(os, 'time') else 12345
            import time
            timestamp = int(time.time())
            filename = f"{name}_{timestamp}.jpg"
            target_path = os.path.join(user_dir, filename)
            
            if image_path:
                import shutil
                shutil.copy(image_path, target_path)
            elif frame is not None:
                cv2.imwrite(target_path, frame)
            else:
                return False, "Không có dữ liệu hình ảnh."

            # Kiểm tra xem có khuôn mặt trong ảnh không
            try:
                # Detector 'opencv' nhanh nhat cho viec kiem tra ton tai mat
                objs = DeepFace.extract_faces(img_path=target_path, detector_backend='opencv', enforce_detection=True)
                if len(objs) > 0:
                    logger.info(f"Đã đăng ký thành công khuôn mặt: {name}")
                    return True, f"Đã đăng ký {name}"
                else:
                    os.remove(target_path)
                    return False, "Không tìm thấy khuôn mặt trong ảnh."
            except Exception as e:
                if os.path.exists(target_path): os.remove(target_path)
                return False, f"Không nhận diện được mặt: {e}"
                
        except Exception as e:
            return False, f"Lỗi hệ thống: {e}"

    def identify_face(self, frame):
        """Nhận diện khuôn mặt trong một frame hình ảnh sử dụng DeepFace.find."""
        try:
            # Luu frame tam thoi de DeepFace xu ly (hoac truyen truc tiep array)
            # Ket qua tra ve mot danh sach cac DataFrame
            results = DeepFace.find(img_path=frame, 
                                    db_path=self.faces_dir, 
                                    model_name=self.model_name, 
                                    detector_backend='opencv', 
                                    enforce_detection=False,
                                    silent=True)
            
            if len(results) > 0 and not results[0].empty:
                # Lay ket qua tot nhat (identity la duong dan file)
                best_match_path = results[0].iloc[0]['identity']
                # Lay ten thu muc cha (ten nguoi dung)
                user_name = os.path.basename(os.path.dirname(best_match_path))
                return user_name
                
            return "Unknown"
        except Exception as e:
            # logger.error(f"Loi khi nhan dien mat: {e}")
            return "Unknown"
