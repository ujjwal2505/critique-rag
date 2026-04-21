"""Per-request JSON trace logging.

Each node appends a structured event to ``state["trace_events"]`` (via the
additive reducer in ``state.py``). After the graph finishes, ``write_trace_file``
dumps the whole run — query used at each step, chunks retrieved, the answer, and
the critic verdict — so the retry loop is auditable in a demo.
"""

import json
import uuid
from datetime import datetime, timezone

from .config import TRACES_DIR


def event(node: str, **data) -> dict:
    """Build a single trace event for a node."""
    return {"node": node, "ts": datetime.now(timezone.utc).isoformat(), **data}


def write_trace_file(final_state: dict) -> str:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    request_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    path = TRACES_DIR / f"{request_id}.json"
    payload = {
        "request_id": request_id,
        "original_query": final_state.get("original_query"),
        "status": final_state.get("status"),
        "final_answer": final_state.get("final_answer"),
        "retry_count": final_state.get("retry_count", 0),
        "max_retries": final_state.get("max_retries"),
        "critic_enabled": final_state.get("critic_enabled"),
        "events": final_state.get("trace_events", []),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)
