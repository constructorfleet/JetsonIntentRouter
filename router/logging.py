import json
import uuid
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

RAW_LOG = LOG_DIR / "raw_requests.jsonl"
ROUTED_LOG = LOG_DIR / "routed_clauses.jsonl"
EXEC_LOG = LOG_DIR / "execution_results.jsonl"


def _write(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_raw_request(utterance: str) -> str:
    rid = uuid.uuid4().hex
    _write(RAW_LOG, {
        "request_id": rid,
        "timestamp": datetime.utcnow().isoformat(),
        "utterance": utterance,
    })
    return rid


def log_routed_clauses(request_id: str, clauses: list):
    _write(ROUTED_LOG, {
        "request_id": request_id,
        "clauses": clauses,
    })


def log_execution_result(request_id: str, results: list):
    _write(EXEC_LOG, {
        "request_id": request_id,
        "results": results,
    })
