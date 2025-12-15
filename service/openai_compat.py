from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List


def extract_last_user_content(messages: List[Dict[str, Any]]) -> str:
    # OpenAI format: list of dicts {role, content}
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                return c
            # If content is array of parts, join text parts
            if isinstance(c, list):
                out = []
                for part in c:
                    if part.get("type") == "text":
                        out.append(part.get("text", ""))
                return "\n".join(out)
    return ""

def chat_completions_response(
        content: str,
        model: str = "router",
        route_meta: Dict[str, Any] | None = None
):
    now = int(time.time())
    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    return {
        "id": rid,
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "route": route_meta or {}
    }
