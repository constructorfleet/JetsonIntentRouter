from intent_router.router.splitter import split_clauses


def test_single_clause():
    text = "turn off the lights"
    clauses = split_clauses(text)
    assert clauses == ["turn off the lights"]


def test_multiple_clauses():
    text = "play alien and turn off the lights"
    clauses = split_clauses(text)
    assert clauses == ["play alien", "turn off the lights"]


def test_then_clause():
    text = "find alien then play it"
    clauses = split_clauses(text)
    assert clauses == ["find alien", "play it"]
