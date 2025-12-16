from intent_router.router.splitter import ClauseSplitter, SplitterConfig

splitter = ClauseSplitter(
    SplitterConfig(
        patterns=["and", "then"],
        min_clause_chars=1,
    )
)

def test_single_clause():
    text = "turn off the lights"
    clauses = splitter.split(text)
    assert clauses == ["turn off the lights"]


def test_multiple_clauses():
    text = "play alien and turn off the lights"
    clauses = split_clauses(text)
    assert clauses == ["play alien", "turn off the lights"]


def test_then_clause():
    text = "find alien then play it"
    clauses = splitter.split(text)
    assert clauses == ["find alien", "play it"]
