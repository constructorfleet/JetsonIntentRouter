from __future__ import annotations

from typing import Any, Dict, Iterator


class Agent:
    def run(self, user_text: str, system_prompt: str | None = None, **kwargs) -> str:
        raise NotImplementedError

    # OPTIONAL
    def stream(
        self, user_text: str, system_prompt: str | None = None, **kwargs
    ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError
