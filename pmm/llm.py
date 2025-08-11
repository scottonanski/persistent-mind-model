import os
import requests


class OpenAIClient:
    def __init__(
        self,
        model: str = None,
        base: str = "https://api.openai.com/v1",
        temp: float = 0.4,
        timeout: int = 40,
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
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 503:
                raise RuntimeError(
                    "OpenAI API temporarily unavailable (503). Try again in a few moments."
                ) from e
            elif r.status_code == 429:
                raise RuntimeError(
                    "OpenAI API rate limit exceeded (429). Wait before retrying."
                ) from e
            elif r.status_code >= 500:
                raise RuntimeError(
                    f"OpenAI API server error ({r.status_code}). Service may be down."
                ) from e
            else:
                raise RuntimeError(
                    f"OpenAI API error ({r.status_code}): {r.text}"
                ) from e
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"OpenAI API request timed out after {self.timeout}s"
            ) from None
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Failed to connect to OpenAI API. Check internet connection."
            ) from None
