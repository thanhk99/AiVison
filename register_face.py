import cv2
import os
import sys
import logging
import time

abs_path = os.path.abspath(__file__)
project_root = os.path.dirname(abs_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from core.config_loader import load_config
from engines.vision.face_manager import FaceManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RegisterFace")

def main():
    print("\n" + "="*40)
    print("CHƯƠNG TRÌNH ĐĂNG KÍ KHUÔN MẶT")
    print("="*40 + "\n")
    
    name = input("Nhập tên người dùng muốn đăng ký: ").strip()
    if not name:
        print("Tên không được để trống!")
        return

    # Tải cấu hình để lấy camera_id
    config = load_config()
    camera_id = config.get("vision", {}).get("camera_id", 0)
    
    face_mgr = FaceManager()
    
    print("Chọn phương thức đăng ký:")
    print("1. Sử dụng Camera")
    print("2. Chọn ảnh từ máy tính")
    choice = input("Nhập lựa chọn (1/2, mặc định 1): ").strip()

    if choice == '2':
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw() # Ẩn cửa sổ chính của tkinter
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh để đăng ký",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        root.destroy()

        if not file_path:
            print("Đã hủy chọn ảnh.")
            return

        print(f"Đang xử lý ảnh: {file_path}...")
        success, message = face_mgr.register_face(name, image_path=file_path)
        if success:
            print(f"Thành công: {message}")
        else:
            print(f"Thất bại: {message}")
        return

    # Mở camera (hỗ trợ cả ID số hoặc link DroidCam)
    if isinstance(camera_id, str) and camera_id.startswith("http"):
        cap = cv2.VideoCapture(camera_id)
    else:
        # Thử mở không có DSHOW trước, nếu lỗi thì thử lại với DSHOW hoặc mặc định
        cap = cv2.VideoCapture(int(camera_id))
        
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở camera với ID: {camera_id}")
        print("Mẹo: Hãy kiểm tra camera_id trong file config.yaml")
        return

    print(f"Đang chuẩn bị camera cho {name} (ID: {camera_id})...")
    print("Hướng dẫn: Nhấn phím 's' để CHỤP ẢNH, nhấn 'q' để THOÁT.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Dang ky: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Nhan 's' de chup", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Register Face", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("Đang xử lý khuôn mặt...")
            success, message = face_mgr.register_face(name, frame=frame)
            if success:
                print(f"Thành công: {message}")
                cv2.putText(frame, "THANH CONG!", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow("Register Face", frame)
                cv2.waitKey(1500)
                break
            else:
                print(f"Thất bại: {message}")
                print("Hãy thử lại, đảm bảo mặt bạn trong khung hình và đủ ánh sáng.")
        elif key == ord('q'):
            print("Đã hủy đăng ký.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Chương trình kết thúc.")

if __name__ == "__main__":
    main()
