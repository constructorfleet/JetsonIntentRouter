import json

import intent_router.router.logging as logging_module
from intent_router.router.router import IntentRouter, RouterConfig
from intent_router.router.splitter import ClauseSplitter, SplitterConfig
from intent_router.router.streaming import agent_can_stream, stream_agent


def test_clause_splitter_filters_and_is_case_insensitive():
    splitter = ClauseSplitter(SplitterConfig(patterns=["and", "THEN"], min_clause_chars=3))

    clauses = splitter.split(" play  song AND go  then a ")

    # "a" is too short and removed; whitespace and casing handled.
    assert clauses == ["play  song", "go"]


def test_clause_splitter_returns_empty_for_blank_text():
    splitter = ClauseSplitter(SplitterConfig(patterns=["and"]))

    assert splitter.split("   ") == []


def test_intent_router_uses_fallback_when_confidence_low():
    splitter = ClauseSplitter(SplitterConfig(patterns=["and"], min_clause_chars=1))

    class LowConfidenceClassifier:
        def predict(self, text):
            # Always below threshold and not mapped
            return "CommandControl", 0.2

    cfg = RouterConfig(
        intents=["CommandControl"],
        intent_to_agent={"Unknown": "fallback-agent"},
        prompt_templates={},
        confidence_threshold=0.5,
    )
    router = IntentRouter(splitter, LowConfidenceClassifier(), cfg)

    result = router.route("turn on the lights and play music")

    assert [c.intent for c in result.clauses] == ["Unknown", "Unknown"]
    assert all(c.agent == "fallback-agent" for c in result.clauses)
    assert result.meta["num_clauses"] == 2


def test_intent_router_handles_empty_splitter_output():
    class EmptySplitter:
        def split(self, text):
            return []

    class EchoClassifier:
        def predict(self, text):
            return "Known", 0.9

    cfg = RouterConfig(
        intents=["Known"],
        intent_to_agent={"Known": "known-agent", "Unknown": "fallback"},
        prompt_templates={},
        confidence_threshold=0.6,
    )
    router = IntentRouter(EmptySplitter(), EchoClassifier(), cfg)

    result = router.route("single clause text")

    assert len(result.clauses) == 1
    assert result.clauses[0].clause == "single clause text"
    assert result.clauses[0].agent == "known-agent"


def test_agent_can_stream_and_stream_agent_fallbacks():
    class StreamingAgent:
        def stream(self, user_text, system_prompt=None):
            yield {"choices": [{"delta": {"content": user_text.upper()}}]}

    class NonStreamingAgent:
        def run(self, user_text, system_prompt=None):
            return f"resp:{user_text}"

    streaming_agent = StreamingAgent()
    non_streaming_agent = NonStreamingAgent()

    assert agent_can_stream(streaming_agent) is True
    assert agent_can_stream(non_streaming_agent) is False

    stream_chunks = list(stream_agent(streaming_agent, user_text="hi", system_prompt="sys"))
    fallback_chunks = list(stream_agent(non_streaming_agent, user_text="hi", system_prompt="sys"))

    assert stream_chunks == [{"choices": [{"delta": {"content": "HI"}}]}]
    assert fallback_chunks == [{"choices": [{"delta": {"content": "resp:hi"}}]}]


def test_logging_writes_json_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_module, "LOG_DIR", tmp_path)
    monkeypatch.setattr(logging_module, "RAW_LOG", tmp_path / "raw_requests.jsonl")
    monkeypatch.setattr(logging_module, "ROUTED_LOG", tmp_path / "routed_clauses.jsonl")
    monkeypatch.setattr(logging_module, "EXEC_LOG", tmp_path / "execution_results.jsonl")

    request_id = logging_module.log_raw_request("hello world")
    logging_module.log_routed_clauses(request_id, [{"intent": "Test"}])
    logging_module.log_execution_result(request_id, [{"status": "ok"}])

    assert len(request_id) == 32  # uuid4 hex

    raw_records = (tmp_path / "raw_requests.jsonl").read_text().splitlines()
    routed_records = (tmp_path / "routed_clauses.jsonl").read_text().splitlines()
    exec_records = (tmp_path / "execution_results.jsonl").read_text().splitlines()

    assert len(raw_records) == len(routed_records) == len(exec_records) == 1
    assert json.loads(raw_records[0])["utterance"] == "hello world"
    assert json.loads(routed_records[0])["clauses"] == [{"intent": "Test"}]
    assert json.loads(exec_records[0])["results"] == [{"status": "ok"}]
