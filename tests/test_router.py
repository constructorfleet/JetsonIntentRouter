from intent_router.router.router import IntentRouter, RouterConfig
from intent_router.router.splitter import (
    ClauseSplitter,
    SplitterConfig,
)

splitter_config = SplitterConfig(
    patterns=["and", "then"],
    min_clause_chars=1,
)

splitter = ClauseSplitter(splitter_config)


class DummyClassifier:
    def predict(self, text):
        return "CommandControl", 0.99


router_config = RouterConfig(
    intents=["CommandControl", "MediaPlayback"],
    intent_to_agent={},
    prompt_templates={},
    confidence_threshold=0.6,
)


def test_router_returns_clauses():
    router = IntentRouter(
        splitter,
        DummyClassifier(),
        router_config,
    )

    result = router.route("turn off the lights and play music")

    assert hasattr(result, "clauses")
    assert len(result.clauses) == 2

    for rc in result.clauses:
        assert hasattr(rc, "clause")
        assert hasattr(rc, "intent")
        assert hasattr(rc, "confidence")
