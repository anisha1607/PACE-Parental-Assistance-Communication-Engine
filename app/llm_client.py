

from __future__ import annotations

import os
import requests



class BaseLLMClient:
    """All providers expose a single .chat() method."""

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.25,
        max_tokens: int = 180,
    ) -> str:
        raise NotImplementedError




class OllamaClient(BaseLLMClient):
    def __init__(self, base_url: str | None = None, timeout: int = 600):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def chat(self, model: str, messages: list, temperature: float = 0.25, max_tokens: int = 180) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]




class OpenAICompatClient(BaseLLMClient):
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: int = 600,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key  # may be None for local servers
        self.timeout = timeout

    def chat(self, model: str, messages: list, temperature: float = 0.25, max_tokens: int = 180) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]




class AnthropicClient(BaseLLMClient):
    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str | None = None, timeout: int = 600):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")
        self.timeout = timeout

    def chat(self, model: str, messages: list, temperature: float = 0.25, max_tokens: int = 180) -> str:
        # Anthropic separates the system prompt from the messages array
        system_content = ""
        filtered_messages = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                filtered_messages.append(m)

        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": filtered_messages,
        }
        if system_content:
            payload["system"] = system_content

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }
        resp = requests.post(self.BASE_URL, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]



# Factory


_PROVIDER_DEFAULTS: dict[str, dict] = {
    "ollama":           {"model": "llama3.1:latest"},
    "groq":             {"model": "llama-3.1-8b-instant"},
    "openai":           {"model": "gpt-4o-mini"},
    "anthropic":        {"model": "claude-haiku-4-5-20251001"},
    "openai_compatible": {"model": ""},
}


def build_client() -> tuple[BaseLLMClient, str]:
    """
    Read LLM_PROVIDER (and related env vars) and return
    (client, model_name).
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()

    # Allow LLM_MODEL to override the default for any provider
    model = os.getenv("LLM_MODEL", _PROVIDER_DEFAULTS.get(provider, {}).get("model", ""))

    if provider == "ollama":
        client = OllamaClient()

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=groq but GROQ_API_KEY is not set.")
        client = OpenAICompatClient(
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=api_key,
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set.")
        client = OpenAICompatClient(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=api_key,
        )

    elif provider == "anthropic":
        client = AnthropicClient()

    elif provider == "openai_compatible":
        base_url = os.getenv("OPENAI_COMPAT_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "LLM_PROVIDER=openai_compatible but OPENAI_COMPAT_BASE_URL is not set."
            )
        client = OpenAICompatClient(
            base_url=base_url,
            api_key=os.getenv("OPENAI_COMPAT_API_KEY"),  # optional
        )

    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            "Choose: ollama, groq, openai, anthropic, openai_compatible"
        )

    if not model:
        raise RuntimeError(
            f"No model specified for provider '{provider}'. "
            "Set LLM_MODEL in your environment."
        )

    return client, model
