from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AgentResponse:
    content: str
    meta: Dict[str, Any]

class Agent:
    def run(self, user_text: str, system_prompt: str | None = None, **kwargs) -> AgentResponse:
        raise NotImplementedError
