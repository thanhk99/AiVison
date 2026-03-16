# AI Home Assistant - Quản gia Thành 🤖

Dự án AI Home Assistant (Quản gia Thành) là một hệ thống trợ lý ảo thông minh, chạy hoàn toàn offline, tích hợp các công nghệ AI tiên tiến để hỗ trợ người dùng thông qua hình ảnh và giọng nói.

## 🌟 Tính năng nổi bật
- **Bảo mật bằng khuôn mặt (Security ID):** Chỉ người dùng đã đăng ký mới có thể kích hoạt trợ lý.
- **Phân tích cảm xúc (Emotion Analysis):** AI hiểu được trạng thái cảm xúc của bạn để đưa ra phản hồi phù hợp.
- **Điều khiển bằng cử chỉ (Gesture Control):** Tương tác không chạm với bộ cử chỉ tay phong phú.
- **Hội thoại tiếng Việt tự nhiên:** Sử dụng mô hình ngôn ngữ lớn (LLM) và giọng nói AI chất lượng cao.

## 📋 Yêu cầu hệ thống
- **Hệ điều hành:** Windows (khuyến nghị).
- **Python:** 3.10 trở lên.
- **Ollama:** Đã cài đặt và đang chạy (Model mặc định: `gemma2:2b`).
- **Phần cứng:** Webcam (hoặc DroidCam) và Microphone.

## 🛠 Hướng dẫn cài đặt

1. **Khởi tạo môi trường (venv):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. **Cài đặt thư viện:**
   ```powershell
   pip install -r requirements.txt
   ```
   *Lưu ý: Bạn có thể cần cài đặt thêm Build Tools cho C++ để cài đặt `pyaudio` hoặc tải file .whl.*

3. **Cấu hình:**
   Mở file `config.yaml` để chỉnh sửa các thông số như `camera_id` hoặc đường dẫn tới các model.

> [!IMPORTANT]
> **Lưu ý về Model:** Các file model nặng (`.onnx`, `.task`) không được bao gồm trong kho lưu trữ Git để giảm dung lượng. Bạn cần tự tải hoặc sao chép các file này vào thư mục `models/` tương ứng (TTS và Vision) trước khi chạy hệ thống.

## 📖 Hướng dẫn sử dụng

### 1. Đăng ký thông tin người dùng
Chạy script sau để hệ thống ghi nhớ khuôn mặt của bạn:
```powershell
python register_face.py
```
- Nhập tên của bạn khi được hỏi.
- Nhìn vào camera và nhấn phím `s` để chụp ảnh đăng ký.

### 2. Khởi chạy hệ thống chính
```powershell
python main.py
```

### 3. Tương tác với Quản gia Thành
Hệ thống hoạt động dựa trên phím bấm và giọng nói:
- ⌨️ Phím **'S'**: Bắt đầu xác thực khuôn mặt. Bạn có tối đa 3 lần thử.
- 🔓 Sau khi xác thực thành công: Bạn có thể bắt đầu trò chuyện bằng giọng nói với Thành.
- ⌨️ Phím **'L'**: Rời khỏi nhà và khóa AI. Thành sẽ chào tạm biệt và ngừng lắng nghe.

### 🖖 Cử chỉ tay (Hỗ trợ khi đã mở khóa)
- 🖐️ **BAN_TAY:** Yêu cầu AI im lặng ngay lập tức.

## 📁 Cấu trúc dự án
- `engines/`: Chứa các bộ xử lý Vision (DeepFace, MediaPipe), Voice (Ollama, Whisper, Piper).
- `models/`: Lưu trữ các file cấu hình và trọng số của mô hình AI.
- `core/`: Các lớp tiện ích quản lý cấu hình và hệ thống.
- `main.py`: File thực thi chính của hệ thống.

---
*Dự án được phát triển bởi các công nghệ AI mã nguồn mở.*
