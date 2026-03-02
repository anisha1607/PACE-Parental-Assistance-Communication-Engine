import os
import requests


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: int = 600):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout

    def chat(self, model: str, messages: list, temperature: float = 0.25, num_predict: int = 180):
        """
        Calls Ollama /api/chat using the correct REST payload.
        - temperature + num_predict must be inside options for Ollama
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }

        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns: { "message": { "role": "...", "content": "..." }, ... }
        return data["message"]["content"]