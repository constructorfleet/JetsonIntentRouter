from __future__ import annotations

from ..agents.base import Agent


def agent_can_stream(agent: Agent) -> bool:
    return callable(getattr(agent, "stream", None))


def stream_agent(agent: Agent, *, user_text: str, system_prompt: str | None):
    if agent_can_stream(agent):
        yield from agent.stream(user_text, system_prompt=system_prompt)
    else:
        content = agent.run(user_text, system_prompt=system_prompt)
        yield {"choices": [{"delta": {"content": content}}]}
