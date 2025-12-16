from __future__ import annotations

import json
import os

import requests

from intent_router.agents.base import Agent, AgentResponse


class OpenAIChatAgent(Agent):
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: int = 60,
    ):
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.timeout_s = timeout_s

    def run(self, user_text: str, system_prompt: str | None = None, **kwargs) -> AgentResponse:
        # If no API key, return a safe stub (useful for offline/dev).
        if not self.api_key:
            return AgentResponse(
                content=f"[stubbed-openai] {user_text}",
                meta={"stubbed": True, "reason": "OPENAI_API_KEY not set"},
            )

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
        }

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return AgentResponse(content=content, meta={"provider": "openai", "model": self.model})

    def stream(self, user_text: str, system_prompt: str | None = None, **kwargs):
        # Stub path for dev / no key
        if not self.api_key:
            yield {"choices": [{"delta": {"content": f"[stubbed-openai] {user_text}"}}]}
            return

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
            "stream": True,
        }

        with requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=self.timeout_s,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                if not line.startswith(b"data: "):
                    continue

                data = line[6:]
                if data == b"[DONE]":
                    return

                yield json.loads(data)
