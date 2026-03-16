import logging
import os
from deepface import DeepFace

logger = logging.getLogger("EmotionAnalyzer")

class EmotionAnalyzer:
    def __init__(self):
        """Khởi tạo bộ nhận diện cảm xúc sử dụng DeepFace."""
        logger.info("Da khoi tao bo nhan dien cam xuc DeepFace.")

    def analyze_emotion(self, frame):
        """
        Phân tích cảm xúc từ khuôn mặt trong frame sử dụng DeepFace.
        """
        try:
            # Analyze trang thai cam xuc
            results = DeepFace.analyze(img_path=frame, 
                                       actions=['emotion'], 
                                       detector_backend='opencv', 
                                       enforce_detection=False,
                                       silent=True)
            
            if len(results) > 0:
                # Lay ket qua dominant_emotion
                return results[0]["dominant_emotion"]
            return "neutral"
        except Exception as e:
            # logger.error(f"Loi khi phan tich cam xuc: {e}")
            return "neutral"

    def translate_emotion(self, emotion_en):
        """Dịch cảm xúc sang tiếng Việt."""
        mapping = {
            "angry": "Tức giận",
            "disgust": "Khó chịu",
            "fear": "Lo sợ",
            "happy": "Vui vẻ",
            "sad": "Buồn bã",
            "surprise": "Ngạc nhiên",
            "neutral": "Bình thường"
        }
        return mapping.get(emotion_en.lower(), "Bình thường")
