"""Officer choices saved on disk. Survive a refresh.

Escalate and Not urgent are different choices, so they are different files.
A message is only in one of them: the latest click replaces the other.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DESK = ROOT / "outputs" / "desk"
FILES = {
    "escalated": DESK / "escalated.json",
    "not_urgent": DESK / "not_urgent.json",
}


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get(case_id: str) -> dict | None:
    for path in FILES.values():
        row = _read(path).get(case_id)
        if isinstance(row, dict):
            return row
    return None


def record(case_id: str, action: str, item: dict) -> dict:
    path = FILES[action]
    entry = {
        "case_id": case_id,
        "action": action,
        "file": path.name,
        "at": datetime.now(ZoneInfo("Australia/Darwin")).strftime("%d %b %Y · %H:%M"),
        "subject": item.get("subject") or "",
        "channel": item.get("channel") or "",
        "score": item.get("score_label") or "",
        "queue": item.get("suggested_queue") or "",
    }
    data = _read(path)
    data[case_id] = entry
    _write(path, data)
    for other_action, other_path in FILES.items():
        if other_action == action:
            continue
        other = _read(other_path)
        if case_id in other:
            del other[case_id]
            _write(other_path, other)
    return entry


def recent(n: int = 8) -> list[dict]:
    rows: list[dict] = []
    for path in FILES.values():
        rows.extend(v for v in _read(path).values() if isinstance(v, dict))
    rows.sort(key=lambda r: r.get("at") or "", reverse=True)
    return rows[:n]
