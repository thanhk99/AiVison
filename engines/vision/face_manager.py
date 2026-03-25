import os
import logging
import cv2
from deepface import DeepFace

logger = logging.getLogger("FaceManager")

class FaceManager:
    def __init__(self, faces_dir="models/vision/faces"):
        self.faces_dir = faces_dir
        self.model_name = "Facenet512" # Model tien tien cho do chinh xac cao
        
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
            
            if frame is not None:
                # Kiem tra anh mo truoc khi xu ly
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                if variance < 80:
                    return False, "Ảnh quá mờ! Vui lòng giữ khuôn mặt tĩnh."
                    
            # Tạm lưu file ảnh
            if image_path:
                import shutil
                shutil.copy(image_path, target_path)
            elif frame is not None:
                cv2.imwrite(target_path, frame)
            else:
                return False, "Không có dữ liệu hình ảnh."

            # Kiểm tra xem có khuôn mặt trong ảnh không
            try:
                # Detector 'mtcnn' chuan xac hon
                objs = DeepFace.extract_faces(img_path=target_path, detector_backend='mtcnn', enforce_detection=True)
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
            # Blur check
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                if variance < 80:
                    logger.warning(f"Anh camera qua mo ({variance:.2f}), bo qua xac thuc.")
                    return "Unknown"

            # Luu frame tam thoi de DeepFace xu ly (hoac truyen truc tiep array)
            # Ket qua tra ve mot danh sach cac DataFrame
            results = DeepFace.find(img_path=frame, 
                                    db_path=self.faces_dir, 
                                    model_name=self.model_name, 
                                    distance_metric='cosine',
                                    detector_backend='mtcnn', 
                                    enforce_detection=True,
                                    silent=True)
            
            if len(results) > 0 and not results[0].empty:
                # Lay ket qua tot nhat (identity la duong dan file)
                best_match = results[0].iloc[0]
                best_match_path = best_match['identity']
                distance = best_match.get('distance', 1.0)
                
                # Check nguong khoang cach that nghang nghiet (Facenet512 ~ 0.30 - 0.23)
                if distance > 0.30:
                    logger.warning(f"Mat giong nhung khoang cach qua lon ({distance:.3f}), tu choi xac thuc.")
                    return "Unknown"
                
                # Lay ten thu muc cha (ten nguoi dung)
                user_name = os.path.basename(os.path.dirname(best_match_path))
                return user_name
                
            return "Unknown"
        except Exception as e:
            # logger.error(f"Loi khi nhan dien mat: {e}")
            return "Unknown"
