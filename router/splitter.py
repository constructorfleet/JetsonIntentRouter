import re
from dataclasses import dataclass
from typing import List


@dataclass
class SplitterConfig:
    patterns: List[str]
    min_clause_chars: int = 2


class ClauseSplitter:
    def __init__(self, cfg: SplitterConfig):
        self.cfg = cfg
        self._pattern = re.compile("|".join(cfg.patterns), flags=re.IGNORECASE)

    def split(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        parts = [p.strip() for p in self._pattern.split(text) if p and p.strip()]
        return [p for p in parts if len(p) >= self.cfg.min_clause_chars]
