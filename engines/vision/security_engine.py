import logging
import cv2
from engines.vision.face_manager import FaceManager
from engines.vision.emotion_analyzer import EmotionAnalyzer

logger = logging.getLogger("SecurityEngine")

class SecurityEngine:
    def __init__(self):
        """
        Khoi tao Security Engine de quan ly xac thuc khuon mat va cam xuc.
        """
        self.face_manager = FaceManager()
        self.emotion_analyzer = EmotionAnalyzer()
        self.authenticated_user = "Unknown"
        self.current_emotion = "Neutral"

    def authenticate(self, frame):
        """
        Xac thuc khuon mat tu frame hinh anh.
        Tra ve ten nguoi dung neu thanh cong, guest neu khong khop, hoac Unknown neu khong thay mat.
        """
        user = self.face_manager.identify_face(frame)
        if user != "Unknown":
            self.authenticated_user = user
            # Neu xac thuc thanh cong, phan tich cam xuc luon
            emo_en = self.emotion_analyzer.analyze_emotion(frame)
            self.current_emotion = self.emotion_analyzer.translate_emotion(emo_en)
            logger.info(f"Xac thuc thanh cong: {user} - Cam xuc: {self.current_emotion}")
        else:
            self.authenticated_user = "Guest"
            self.current_emotion = "N/A"
            logger.warning("Nguoi la dang truy cap.")
            
        return self.authenticated_user

    def reset(self):
        """Reset trang thai xac thuc."""
        self.authenticated_user = "Unknown"
        self.current_emotion = "Neutral"

    def get_user_info(self):
        return {
            "user": self.authenticated_user,
            "emotion": self.current_emotion
        }
