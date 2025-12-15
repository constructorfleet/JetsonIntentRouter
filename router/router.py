from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from router.splitter import ClauseSplitter
from router.types import RoutedClause, RouteResult


@dataclass
class RouterConfig:
    intents: List[str]
    intent_to_agent: Dict[str, str]
    prompt_templates: Dict[str, str]
    confidence_threshold: float = 0.6

class IntentRouter:
    def __init__(self, splitter: ClauseSplitter, classifier, cfg: RouterConfig):
        self.splitter = splitter
        self.classifier = classifier  # must expose: predict(text)->(intent, confidence)
        self.cfg = cfg

    def route(self, text: str) -> RouteResult:
        clauses = self.splitter.split(text)
        routed: List[RoutedClause] = []

        for clause in clauses or [text]:
            intent, conf = self.classifier.predict(clause)
            if conf < self.cfg.confidence_threshold or intent not in self.cfg.intent_to_agent:
                intent = "Unknown"
            agent = self.cfg.intent_to_agent.get(
                    intent,
                    self.cfg.intent_to_agent.get("Unknown", "search")
            )
            routed.append(RoutedClause(
                    clause=clause,
                    intent=intent,
                    confidence=float(conf),
                    agent=agent)
            )

        return RouteResult(original_text=text, clauses=routed, meta={"num_clauses": len(routed)})
