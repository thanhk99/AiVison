import requests
import json
import logging

logger = logging.getLogger("LLMEngine")

class LLMEngine:
    def __init__(self, base_url="http://localhost:6011", model="deepseek-v4-flash-nothinking", api_key="thanh2004"):
        """
        Khởi tạo kết nối tới ds2api (OpenAI compatible API).
        """
        # Endpoint chuẩn của OpenAI compatible
        self.base_url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.model = model
        self.api_key = api_key
        self.system_prompt = ""
        self.history = []  # Lịch sử hội thoại
        self.max_history = 10  # Tối đa 5 cặp

    def _build_payload(self, stream: bool) -> dict:
        """Tạo payload gửi tới API, bao gồm lịch sử và system prompt."""
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "response_format": {"type": "json_object"},  # Yêu cầu trả về JSON
            "temperature": 0.1  # Thấp để AI tuân thủ định dạng
        }

    def _get_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_response_full(self, user_input: str) -> str:
        """
        Gửi yêu cầu tới LLM và nhận toàn bộ response (không stream).
        Phù hợp để parse JSON sau đó.
        """
        self.history.append({"role": "user", "content": user_input})
        payload = self._build_payload(stream=False)

        try:
            response = requests.post(self.base_url, headers=self._get_headers(), json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            full_response = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

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
        payload = self._build_payload(stream=True)

        full_response = ""
        try:
            response = requests.post(self.base_url, headers=self._get_headers(), json=payload, stream=True, timeout=30)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith("data: "):
                        decoded = decoded[6:]
                    if decoded == "[DONE]":
                        break
                    if decoded:
                        chunk = json.loads(decoded)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
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
