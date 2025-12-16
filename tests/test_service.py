import json

import pytest
from flask import Flask

import intent_router.service.app as app_module
import intent_router.service.config as config_module
import intent_router.service.labeling as labeling_module
from intent_router.service.openai_compat import (
    chat_completions_response,
    extract_last_user_content,
)


def test_load_service_config_env(monkeypatch):
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "1234")
    monkeypatch.setenv("ROUTER_BACKEND", "trt")
    monkeypatch.setenv("ROUTER_MODEL_PATH", "model.plan")
    monkeypatch.setenv("ROUTER_INTENTS_PATH", "i.yaml")
    monkeypatch.setenv("ROUTER_SPLITTER_PATH", "s.yaml")
    monkeypatch.setenv("ROUTER_AGENTS_PATH", "a.yaml")
    monkeypatch.setenv("ROUTER_SEQ_LEN", "16")

    cfg = config_module.load_service_config()

    assert cfg.host == "0.0.0.0"
    assert cfg.port == 1234
    assert cfg.backend == "trt"
    assert cfg.model_path == "model.plan"
    assert cfg.intents_path == "i.yaml"
    assert cfg.splitter_path == "s.yaml"
    assert cfg.agents_path == "a.yaml"
    assert cfg.seq_len == 16


def test_load_router_bits_reads_yaml(tmp_path):
    intents_file = tmp_path / "intents.yaml"
    intents_file.write_text("intents:\\n  - A\\n  - B\\n")
    splitter_file = tmp_path / "splitter.yaml"
    splitter_file.write_text("patterns:\\n  - and\\nmin_clause_chars: 5\\n")
    agents_file = tmp_path / "agents.yaml"
    agents_file.write_text(
        "intent_to_agent:\\n  A: agent_a\\nprompt_templates:\\n  A: '{clause}!'\\nconfidence_threshold: 0.8\\nagents:\\n  agent_a:\\n    type: local_command\\n"
    )

    intents, split_cfg, agents_cfg, router_cfg = config_module.load_router_bits(
        intents_file, splitter_file, agents_file
    )

    assert intents == ["A", "B"]
    assert split_cfg.patterns == ["and"]
    assert split_cfg.min_clause_chars == 5
    assert router_cfg.intent_to_agent == {"A": "agent_a"}
    assert router_cfg.prompt_templates == {"A": "{clause}!"}
    assert router_cfg.confidence_threshold == 0.8
    assert agents_cfg["agents"]["agent_a"]["type"] == "local_command"


def test_build_agents_creates_known_types_and_raises(monkeypatch):
    cfg = {
        "agents": {
            "chatty": {"type": "openai_chat"},
            "local": {"type": "local_command"},
        }
    }

    agents = app_module.build_agents(cfg)

    assert set(agents.keys()) == {"chatty", "local"}
    assert agents["chatty"].__class__.__name__ == "OpenAIChatAgent"
    assert agents["local"].__class__.__name__ == "LocalCommandAgent"

    bad_cfg = {"agents": {"x": {"type": "unknown"}}}
    with pytest.raises(ValueError):
        app_module.build_agents(bad_cfg)


def test_execute_clauses_streams_and_records_success(monkeypatch):
    class RecordingAgent:
        def __init__(self):
            self.calls = []

        def stream(self, user_text, system_prompt=None):
            self.calls.append((user_text, system_prompt))
            yield {"choices": [{"delta": {"content": f"resp:{user_text}"}}]}

    agents_cfg = {
        "agents": {"agent1": {"type": "local_command", "system_prompt": "sys"}},
        "prompt_templates": {"Intent1": "tmpl {clause}"},
    }
    agents = {"agent1": RecordingAgent()}

    class RC:
        def __init__(self):
            self.clause = "do it"
            self.intent = "Intent1"
            self.confidence = 0.9
            self.agent = "agent1"

    route_result = type("RR", (), {"clauses": [RC()]})

    gen = app_module.execute_clauses(route_result, agents, agents_cfg)
    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as stop:
        exec_results = stop.value

    assert chunks == [{"choices": [{"delta": {"content": "resp:tmpl do it"}}]}]
    assert exec_results == [
        {"clause": "do it", "agent": "agent1", "status": "success"},
    ]
    assert agents["agent1"].calls == [("tmpl do it", "sys")]


def test_format_routed_output_string():
    class RC:
        def __init__(self, clause, intent, conf, agent):
            self.clause = clause
            self.intent = intent
            self.confidence = conf
            self.agent = agent

    rr = type(
        "RR",
        (),
        {
            "clauses": [
                RC("c1", "I1", 0.9, "a1"),
                RC("c2", "I2", 0.5, "a2"),
            ]
        },
    )
    out = app_module.format_routed_output(rr, ["ok1", "ok2"])
    assert "Routed clauses:" in out
    assert "- clause: c1" in out
    assert "intent: I2" in out
    assert "result: ok2" in out


def test_extract_last_user_content_handles_structured():
    messages = [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]},
        {"role": "user", "content": "final"},
    ]
    assert extract_last_user_content(messages) == "final"
    assert extract_last_user_content(messages[:-1]) == "part1\npart2"
    assert extract_last_user_content([]) == ""


def test_chat_completions_response_shape():
    resp = chat_completions_response("hello", model="x", route_meta={"a": 1})
    assert resp["object"] == "chat.completion"
    assert resp["choices"][0]["message"]["content"] == "hello"
    assert resp["route"] == {"a": 1}
    assert resp["model"] == "x"
    assert resp["usage"]["total_tokens"] == 0


def test_load_unreviewed_reads_jsonl(tmp_path, monkeypatch):
    log_file = tmp_path / "routed.jsonl"
    log_file.write_text(json.dumps({"a": 1}) + "\n" + json.dumps({"b": 2}) + "\n")
    monkeypatch.setattr(labeling_module, "ROUTED_LOG", log_file)

    rows = labeling_module.load_unreviewed(limit=1)
    assert rows == [{"a": 1}]


def test_label_post_writes_accepts(monkeypatch, tmp_path):
    monkeypatch.setattr(labeling_module, "OUT_FILE", tmp_path / "accepted.jsonl")
    monkeypatch.setattr(labeling_module, "ROUTED_LOG", tmp_path / "routed.jsonl")
    monkeypatch.setattr(labeling_module, "render_template", lambda *_, **__: "")

    app = Flask(__name__)
    app.register_blueprint(labeling_module.LABEL_UI)
    client = app.test_client()

    resp = client.post("/label", data={"text": "hello", "label": "Media"})

    assert resp.status_code in (301, 302)
    data = (tmp_path / "accepted.jsonl").read_text().strip().splitlines()
    assert data == [json.dumps({"text": "hello", "label": "Media"})]
