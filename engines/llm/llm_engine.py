import requests
import json
import logging

logger = logging.getLogger("LLMEngine")

class LLMEngine:
    def __init__(self, base_url="http://localhost:11434", model="gemma2:2b"):
        """
        Khởi tạo kết nối tới Ollama API.
        """
        self.base_url = f"{base_url}/api/chat" # Chuyển sang endpoint chat để giữ ngữ cảnh
        self.model = model
        self.system_prompt = ""
        self.history = [] # Lưu lịch sử hội thoại: [{"role": "user", "content": "..."}, ...]
        self.max_history = 10 # Giữ tối đa 5 cặp hội thoại (10 dòng)

    def generate_response_stream(self, user_input: str):
        """
        Gửi hội thoại tới LLM (kèm lịch sử) và nhận phản hồi dạng stream.
        """
        # Thêm câu mới của người dùng vào lịch sử
        self.history.append({"role": "user", "content": user_input})
        
        # Chuẩn bị payload kèm System Prompt và Lịch sử
        messages = [{"role": "system", "content": self.system_prompt}] + self.history

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0 # Giảm độ sáng tạo để AI tuân thủ luật cấm icon
            }
        }
        
        full_response = ""
        try:
            response = requests.post(self.base_url, json=payload, stream=True, timeout=30)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if not chunk.get("done"):
                        content = chunk.get("message", {}).get("content", "")
                        full_response += content
                        yield content
            
            # Lưu câu trả lời của AI vào lịch sử sau khi hoàn tất
            self.history.append({"role": "assistant", "content": full_response})
            
            # Cắt bớt lịch sử nếu quá dài
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
                
        except Exception as e:
            yield f"\n[Lỗi LLM] {str(e)}"
    
    def clear_history(self):
        self.history = []
