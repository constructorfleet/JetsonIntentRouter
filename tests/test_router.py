from router.router import Router


class DummyClassifier:
    def predict(self, text):
        return "CommandControl", 0.99


def test_router_returns_clauses():
    router = Router(classifier=DummyClassifier())

    result = router.route("turn off the lights and play music")

    assert hasattr(result, "clauses")
    assert len(result.clauses) == 2

    for rc in result.clauses:
        assert hasattr(rc, "clause")
        assert hasattr(rc, "intent")
        assert hasattr(rc, "confidence")
