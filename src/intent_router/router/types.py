from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RoutedClause:
    clause: str
    intent: str
    confidence: float
    agent: str


@dataclass
class RouteResult:
    original_text: str
    clauses: List[RoutedClause]
    meta: Dict[str, Any]
