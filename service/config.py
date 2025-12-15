from __future__ import annotations
import os, yaml
from dataclasses import dataclass
from router.splitter import SplitterConfig
from router.router import RouterConfig

@dataclass
class ServiceConfig:
    host: str
    port: int
    backend: str
    model_path: str
    intents_path: str
    splitter_path: str
    agents_path: str
    seq_len: int

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_service_config() -> ServiceConfig:
    return ServiceConfig(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        backend=os.getenv("ROUTER_BACKEND", "ort"),
        model_path=os.getenv("ROUTER_MODEL_PATH", "models/intent_classifier_sim.onnx"),
        intents_path=os.getenv("ROUTER_INTENTS_PATH", "config/intents.yaml"),
        splitter_path=os.getenv("ROUTER_SPLITTER_PATH", "config/splitter.yaml"),
        agents_path=os.getenv("ROUTER_AGENTS_PATH", "config/agents.yaml"),
        seq_len=int(os.getenv("ROUTER_SEQ_LEN", "32")),
    )

def load_router_bits(intents_path: str, splitter_path: str, agents_path: str):
    intents_cfg = load_yaml(intents_path)
    splitter_cfg = load_yaml(splitter_path)
    agents_cfg = load_yaml(agents_path)

    intents = intents_cfg["intents"]
    split_cfg = SplitterConfig(
        patterns=splitter_cfg["patterns"],
        min_clause_chars=int(splitter_cfg.get("min_clause_chars", 2))
    )

    router_cfg = RouterConfig(
        intents=intents,
        intent_to_agent=agents_cfg["intent_to_agent"],
        prompt_templates=agents_cfg.get("prompt_templates", {}),
        confidence_threshold=float(agents_cfg.get("confidence_threshold", 0.6))
    )
    return intents, split_cfg, agents_cfg, router_cfg
