from __future__ import annotations

import json

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, stream_with_context

from intent_router.agents.local_command import LocalCommandAgent
from intent_router.agents.openai_chat import OpenAIChatAgent
from intent_router.router.logging import (
    log_execution_result,
    log_raw_request,
    log_routed_clauses,
)
from intent_router.router.router import IntentRouter
from intent_router.router.splitter import ClauseSplitter
from intent_router.router.streaming import stream_agent
from intent_router.runtime.ort_classifier import OrtIntentClassifier
from intent_router.service.config import load_router_bits, load_service_config
from intent_router.service.labeling import LABEL_UI
from intent_router.service.openai_compat import chat_completions_response, extract_last_user_content


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


def execute_clauses(route_result, agents, agents_cfg):
    exec_results = []

    for rc in route_result.clauses:
        agent = agents[rc.agent]
        agent_cfg = agents_cfg["agents"][rc.agent]
        sys_prompt = agent_cfg.get("system_prompt")

        tmpl = agents_cfg.get("prompt_templates", {}).get(rc.intent, "{clause}")
        user_prompt = tmpl.format(clause=rc.clause)

        try:
            for chunk in stream_agent(
                agent,
                user_text=user_prompt,
                system_prompt=sys_prompt,
            ):
                yield chunk

            exec_results.append(
                {
                    "clause": rc.clause,
                    "agent": rc.agent,
                    "status": "success",
                }
            )

        except Exception as e:
            exec_results.append(
                {
                    "clause": rc.clause,
                    "agent": rc.agent,
                    "status": "error",
                    "error": str(e),
                }
            )
            raise

    return exec_results


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

    intents, split_cfg, agents_cfg, router_cfg = load_router_bits(
        cfg.intents_path, cfg.splitter_path, cfg.agents_path
    )
    splitter = ClauseSplitter(split_cfg)
    classifier = build_classifier(cfg.backend, cfg.model_path, intents, cfg.seq_len)
    router = IntentRouter(splitter=splitter, classifier=classifier, cfg=router_cfg)
    agents = build_agents(agents_cfg)

    app = Flask(__name__)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "backend": cfg.backend}

    app.register_blueprint(LABEL_UI)

    # OpenAI-compatible endpoint (minimal): /v1/chat/completions
    @app.post("/v1/chat/completions")
    def chat_completions():
        body = request.get_json(force=True)
        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))

        user_text = extract_last_user_content(messages)

        # ---- logging: raw input ----
        request_id = log_raw_request(user_text)

        # ---- routing ----
        route_result = router.route(user_text)

        # ---- logging: routing ----
        log_routed_clauses(
            request_id,
            [
                {
                    "clause": rc.clause,
                    "intent": rc.intent,
                    "confidence": rc.confidence,
                    "agent": rc.agent,
                }
                for rc in route_result.clauses
            ],
        )

        # ---- streaming path ----
        if stream:

            def event_stream():
                exec_results = []

                for chunk in execute_clauses(route_result, agents, agents_cfg):
                    yield f"data: {json.dumps(chunk)}\n\n"

                # log execution after all clauses
                log_execution_result(request_id, exec_results)

                # ONE termination
                yield f"""data: {json.dumps({
                    'choices': [{
                        'delta': {},
                        'finish_reason': 'stop'
                    }]
                })}\n\n"""
                yield "data: [DONE]\n\n"

            return Response(
                stream_with_context(event_stream()),
                mimetype="text/event-stream",
            )

        # ---- non-streaming path ----
        else:
            collected = []
            exec_results = []

            for chunk in execute_clauses(route_result, agents, agents_cfg):
                # collapse deltas into text
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    collected.append(delta["content"])

            log_execution_result(request_id, exec_results)

            return jsonify(
                chat_completions_response(
                    content="".join(collected),
                    model=body.get("model", "router"),
                )
            )


if __name__ == "__main__":
    app = create_app()
    cfg = load_service_config()
    app.run(host=cfg.host, port=cfg.port, debug=True)
