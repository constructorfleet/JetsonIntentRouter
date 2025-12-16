from __future__ import annotations

from typing import Iterator

from intent_router.agents.types import AgentResponse


class Agent:
    def run(self, user_text: str, system_prompt: str | None = None, **kwargs) -> AgentResponse:
        raise NotImplementedError

    # OPTIONAL
    def stream(
        self, user_text: str, system_prompt: str | None = None, **kwargs
    ) -> Iterator[AgentResponse]:
        raise NotImplementedError
