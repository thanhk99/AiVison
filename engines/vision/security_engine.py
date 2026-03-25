import logging
import cv2
from engines.vision.face_manager import FaceManager

logger = logging.getLogger("SecurityEngine")

class SecurityEngine:
    def __init__(self):
        """
        Khoi tao Security Engine de quan ly xac thuc khuon mat.
        """
        self.face_manager = FaceManager()
        self.authenticated_user = "Unknown"

    def authenticate(self, frame):
        """
        Xac thuc khuon mat tu frame hinh anh.
        Tra ve ten nguoi dung neu thanh cong, guest neu khong khop, hoac Unknown neu khong thay mat.
        """
        user = self.face_manager.identify_face(frame)
        if user != "Unknown":
            self.authenticated_user = user
            logger.info(f"Xac thuc thanh cong: {user}")
        else:
            self.authenticated_user = "Guest"
            logger.debug("Nguoi la dang truy cap hoac khong nhan dien ra ai.")
            
        return self.authenticated_user

    def reset(self):
        """Reset trang thai xac thuc."""
        self.authenticated_user = "Unknown"

    def get_user_info(self):
        return {
            "user": self.authenticated_user
        }
