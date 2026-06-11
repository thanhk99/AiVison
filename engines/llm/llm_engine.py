import json
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Nạp các biến môi trường từ file .env

logger = logging.getLogger("LLMEngine")

class LLMEngine:
    def __init__(self, base_url="https://api.deepseek.com", model="deepseek-chat", api_key=None):
        """
        Khởi tạo kết nối tới DeepSeek API (hoặc OpenAI compatible API) sử dụng thư viện OpenAI SDK.
        """
        # Nếu không truyền api_key hoặc truyền giá trị mặc định của config, lấy từ file .env
        if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY" or api_key == "thanh2004":
            api_key = os.getenv("API_KEY")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.system_prompt = ""
        self.history = []  # Lịch sử hội thoại
        self.max_history = 10  # Tối đa 5 cặp

    def _build_messages(self) -> list:
        """Tạo danh sách messages gửi tới API, bao gồm lịch sử và system prompt."""
        return [{"role": "system", "content": self.system_prompt}] + self.history

    def generate_response_full(self, user_input: str) -> str:
        """
        Gửi yêu cầu tới LLM và nhận toàn bộ response (không stream).
        Phù hợp để parse JSON sau đó.
        """
        self.history.append({"role": "user", "content": user_input})
        messages = self._build_messages()
        
        # Log ra câu hỏi đang chuẩn bị gửi cho AI
        logger.info(f"[LLM] Đang gửi câu hỏi lên AI: {user_input}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            full_response = response.choices[0].message.content.strip()

            # Lưu lại lịch sử
            self.history.append({"role": "assistant", "content": full_response})

            # Cắt bớt lịch sử nếu quá dài
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

            return full_response

        except Exception as e:
            logger.error(f"Lỗi LLM (full): {e}")
            return json.dumps({"message": "Xin lỗi, tôi gặp sự cố kết nối.", "command": "NONE"})

    def generate_response_stream(self, user_input: str):
        """
        Gửi yêu cầu tới LLM và nhận phản hồi dạng stream.
        Dùng cho các trường hợp cần phản hồi nhanh (không cần parse JSON).
        """
        self.history.append({"role": "user", "content": user_input})
        messages = self._build_messages()

        full_response = ""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                response_format={"type": "json_object"},
                temperature=0.1
            )

            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content

            # Lưu lại lịch sử
            self.history.append({"role": "assistant", "content": full_response})

            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

        except Exception as e:
            yield f"\n[Lỗi LLM] {str(e)}"

    def clear_history(self):
        self.history = []
