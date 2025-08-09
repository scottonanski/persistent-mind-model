import os
import requests

class OpenAIClient:
    def __init__(
        self,
        model: str = None,
        base: str = "https://api.openai.com/v1",
        temp: float = 0.4,
        timeout: int = 40
    ):
        # Allow override from environment
        env_model = os.environ.get("OPENAI_MODEL")
        self.model = model or env_model or "gpt-4o-mini"

        self.base = base
        self.temp = temp
        self.timeout = timeout

        key = os.environ.get("OPENAI_API_KEY")
        # allow init without key; usage will error in chat()
        self.key = key if key else None

    def chat(self, system: str, user: str) -> str:
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot run reflection.")
        url = f"{self.base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": self.temp,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
