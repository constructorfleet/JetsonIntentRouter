from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class AgentResponse:
    content: str
    meta: dict[str, Any]


class Agent:
    def run(self, user_text: str, system_prompt: str | None = None, **kwargs) -> AgentResponse:
        raise NotImplementedError

    # OPTIONAL
    def stream(
        self, user_text: str, system_prompt: str | None = None, **kwargs
    ) -> Iterator[AgentResponse]:
        raise NotImplementedError
