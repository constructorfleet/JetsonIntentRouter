from intent_router.router.logging import (
    log_execution_result,
    log_raw_request,
    log_routed_clauses,
)


def test_logging_functions(tmp_path, monkeypatch):
    # Redirect logs to temp dir
    monkeypatch.setattr("router.logging.LOG_DIR", tmp_path)
    monkeypatch.setattr("router.logging.RAW_LOG", tmp_path / "raw.jsonl")
    monkeypatch.setattr("router.logging.ROUTED_LOG", tmp_path / "routed.jsonl")
    monkeypatch.setattr("router.logging.EXEC_LOG", tmp_path / "exec.jsonl")

    rid = log_raw_request("turn off the lights")
    assert rid

    log_routed_clauses(
        rid,
        [{"clause": "turn off the lights", "intent": "CommandControl", "confidence": 0.9}],
    )

    log_execution_result(
        rid,
        [{"clause": "turn off the lights", "status": "success"}],
    )

    assert (tmp_path / "raw.jsonl").exists()
    assert (tmp_path / "routed.jsonl").exists()
    assert (tmp_path / "exec.jsonl").exists()
