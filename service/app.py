from __future__ import annotations
import os
from flask import Flask, request, jsonify, stream_with_context
import json
from dotenv import load_dotenv

from router.splitter import ClauseSplitter
from router.router import IntentRouter
from router.streaming import stream_agent
from runtime.ort_classifier import OrtIntentClassifier
from service.config import load_service_config, load_router_bits
from service.openai_compat import extract_last_user_content, chat_completions_response

from agents.openai_chat import OpenAIChatAgent
from agents.local_command import LocalCommandAgent

def build_classifier(backend: str, model_path: str, intents, seq_len: int):
    if backend == "trt":
        from runtime.trt_classifier import TrtIntentClassifier
        return TrtIntentClassifier(model_path, intents=intents, seq_len=seq_len)
    # default ORT
    return OrtIntentClassifier(model_path, intents=intents, seq_len=seq_len)

def build_agents(agents_cfg):
    agents = {}
    for name, cfg in agents_cfg["agents"].items():
        t = cfg["type"]
        if t == "openai_chat":
            agents[name] = OpenAIChatAgent()
        elif t == "local_command":
            agents[name] = LocalCommandAgent()
        else:
            raise ValueError(f"Unknown agent type: {t} for agent {name}")
    return agents

def format_routed_output(route_result, agent_outputs):
    # Human-readable output returned as assistant content (OpenAI format).
    lines = []
    lines.append("Routed clauses:")
    for rc, out in zip(route_result.clauses, agent_outputs):
        lines.append(f"- clause: {rc.clause}")
        lines.append(f"  intent: {rc.intent} (conf={rc.confidence:.3f})")
        lines.append(f"  agent:  {rc.agent}")
        lines.append(f"  result: {out}")
    return "\n".join(lines)

def create_app():
    load_dotenv()
    cfg = load_service_config()

    intents, split_cfg, agents_cfg, router_cfg = load_router_bits(cfg.intents_path, cfg.splitter_path, cfg.agents_path)
    splitter = ClauseSplitter(split_cfg)
    classifier = build_classifier(cfg.backend, cfg.model_path, intents, cfg.seq_len)
    router = IntentRouter(splitter=splitter, classifier=classifier, cfg=router_cfg)
    agents = build_agents(agents_cfg)

    app = Flask(__name__)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "backend": cfg.backend}

    # OpenAI-compatible endpoint (minimal): /v1/chat/completions
    @app.post("/v1/chat/completions")
    def chat_completions():
        body = request.get_json(force=True)
        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))

        user_text = extract_last_user_content(messages)
        route_result = router.route(user_text)

        if not stream:
            # Optional: keep your existing non-streaming behavior
            return jsonify(chat_completions_response(
                content="Streaming disabled; enable stream=true",
                model=body.get("model", "router"),
            ))

        def event_stream():
            for rc in route_result.clauses:
                agent = agents[rc.agent]
                agent_cfg = agents_cfg["agents"][rc.agent]
                sys_prompt = agent_cfg.get("system_prompt")

                tmpl = agents_cfg.get("prompt_templates", {}).get(rc.intent, "{clause}")
                user_prompt = tmpl.format(clause=rc.clause)

                for chunk in stream_agent(
                    agent,
                    user_text=user_prompt,
                    system_prompt=sys_prompt,
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"

            # ONE termination
            yield f"data: {json.dumps({
                'choices': [{
                    'delta': {},
                    'finish_reason': 'stop'
                }]
            })}\n\n"
            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(event_stream()),
            mimetype="text/event-stream",
        )

    return app

if __name__ == "__main__":
    app = create_app()
    cfg = load_service_config()
    app.run(host=cfg.host, port=cfg.port, debug=True)
