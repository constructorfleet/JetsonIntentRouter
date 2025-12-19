import pytest

import intent_router.agents.openai_chat as openai_chat_module
from intent_router.agents.local_command import LocalCommandAgent
from intent_router.agents.openai_chat import OpenAIChatAgent


def test_local_command_agent_returns_payload():
    agent = LocalCommandAgent()

    response = agent.run("turn on the lights")

    assert response.content.startswith("[local-command]")
    assert response.meta["command"]["type"] == "command_control"
    assert response.meta["command"]["text"] == "turn on the lights"


def test_openai_chat_agent_stubbed_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = OpenAIChatAgent(api_key="")

    response = agent.run("hello world")

    assert "[stubbed-openai] hello world" == response.content
    assert response.meta["stubbed"] is True
    assert "OPENAI_API_KEY not set" in response.meta["reason"]


def test_openai_chat_agent_run_posts_payload(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "hi from openai"}}]}

        return FakeResponse()

    monkeypatch.setattr(openai_chat_module.requests, "post", fake_post)
    agent = OpenAIChatAgent(
        base_url="https://example.com/api/",
        api_key="test-key",
        model="unit-test-model",
        timeout_s=5,
    )

    response = agent.run("ping", system_prompt="system prompt", temperature=0.7)

    assert response.content == "hi from openai"
    assert response.meta == {"provider": "openai", "model": "unit-test-model"}
    assert captured["url"] == "https://example.com/api/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["timeout"] == 5
    assert captured["json"]["model"] == "unit-test-model"
    assert captured["json"]["messages"][0] == {"role": "system", "content": "system prompt"}
    assert captured["json"]["messages"][1] == {"role": "user", "content": "ping"}
    assert captured["json"]["temperature"] == 0.7


def test_openai_chat_agent_stream_stub(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = OpenAIChatAgent(api_key="")

    chunks = list(agent.stream("stream this"))

    assert chunks == [{"choices": [{"delta": {"content": "[stubbed-openai] stream this"}}]}]


def test_openai_chat_agent_stream_yields_chunks(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = {}

    class FakeStreamResponse:
        def __init__(self, lines):
            self.lines = lines

        def raise_for_status(self):
            return None

        def iter_lines(self):
            for line in self.lines:
                yield line

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_post(url, headers, json, stream, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["stream"] = stream
        captured["timeout"] = timeout
        lines = [
            b"",
            b"event: ignore",
            b"data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}",
            b"data: {\"choices\": [{\"delta\": {\"content\": \" World\"}}]}",
            b"data: [DONE]",
        ]
        return FakeStreamResponse(lines)

    monkeypatch.setattr(openai_chat_module.requests, "post", fake_post)
    agent = OpenAIChatAgent(
        base_url="https://example.com/api/",
        api_key="secret",
        model="stream-model",
        timeout_s=9,
    )

    chunks = list(agent.stream("hi there", system_prompt="sys"))

    assert captured["url"] == "https://example.com/api/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["stream"] is True
    assert captured["timeout"] == 9
    assert captured["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["json"]["messages"][1] == {"role": "user", "content": "hi there"}
    assert captured["json"]["stream"] is True
    assert [chunk["choices"][0]["delta"]["content"] for chunk in chunks] == [
        "Hello",
        " World",
    ]
