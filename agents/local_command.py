from __future__ import annotations

from agents.base import Agent, AgentResponse


class LocalCommandAgent(Agent):
    """Stub agent: translate to a pretend command payload."""

    def run(self, user_text: str, system_prompt: str | None = None, **kwargs) -> AgentResponse:
        # Replace this with Home Assistant / MQTT / whatever you actually use.
        cmd = {"type": "command_control", "text": user_text}
        return AgentResponse(content=f"[local-command] {cmd}", meta={"command": cmd})
