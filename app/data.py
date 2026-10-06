"""Load a small officer desk from locked scores + quoted spans.

Hard rules (do not break):
- The model assists. A human decides.
- A miss costs more than an extra alarm.
- One cutoff (0.25) for every style and community.
- Never delete or hide unflagged mail.
- community is never a score
- reasons quote the writer
- Do not send a reply on serious items
"""

from __future__ import annotations

import html
import json
import re
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.explain import path_copy
from app.model import predict_frame

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "public_contact_processed.csv"
EXPLAIN = {
    "val": ROOT / "outputs" / "exports" / "officer_explain_val.csv",
    "test": ROOT / "outputs" / "exports" / "officer_explain_test.csv",
    "redteam": ROOT / "outputs" / "exports" / "officer_explain_redteam.csv",
}
SCORES = {
    "val": ROOT / "outputs" / "reports" / "baseline_val_scores.csv",
    "test": ROOT / "outputs" / "exports" / "score_test.csv",
    "redteam": ROOT / "outputs" / "exports" / "score_redteam.csv",
}
DISTIL_SCORES = ROOT / "outputs" / "reports" / "distilbert_val_scores.csv"
DISTIL_CUTOFF = 0.25

# Pitch path: SMS P1 first, then the URGENT trap still in the list.
DEMO_SMS_P1 = "MSG-000250"
DEMO_URGENT_TRAP = "MSG-000031"
KNOWN_VAL_MISS = "MSG-028158"
MUST_IDS = [
    DEMO_SMS_P1,
    "MSG-025490",
    "MSG-000638",
    "MSG-000465",
    "MSG-000224",
    "MSG-000584",
    KNOWN_VAL_MISS,
    DEMO_URGENT_TRAP,
    "MSG-000003",
]
CUTOFF = 0.38

QUEUE_LABEL = {
    "child_protection": "Child safety",
    "housing": "Housing",
    "health": "Health",
    "infrastructure_roads": "Roads / works",
    "general_complaints": "General",
    "licences_rego": "Licences",
    "concessions_payments": "Payments",
}

_CACHE: list[dict] | None = None


def _parse_spans(raw: object) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [str(s) for s in raw]
    try:
        val = json.loads(str(raw))
        return [str(s) for s in val] if isinstance(val, list) else []
    except json.JSONDecodeError:
        return []


def highlight(text: str, spans: list[str]) -> str:
    """Wrap their words in <mark>. Do not rewrite."""
    text = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    intervals: list[tuple[int, int]] = []
    low = text.casefold()
    for span in sorted({s for s in spans if s}, key=len, reverse=True):
        needle = span.casefold()
        start = 0
        while True:
            i = low.find(needle, start)
            if i < 0:
                break
            j = i + len(span)
            if not any(i < b and a < j for a, b in intervals):
                intervals.append((i, j))
            start = i + max(len(span), 1)
    intervals.sort()
    out: list[str] = []
    last = 0
    for a, b in intervals:
        out.append(html.escape(text[last:a]))
        out.append("<mark>" + html.escape(text[a:b]) + "</mark>")
        last = b
    out.append(html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>\n")


def _when(raw: object) -> tuple[str, str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "", ""
    try:
        dt = pd.to_datetime(raw, utc=True)
        local = dt.tz_convert("Australia/Darwin")
        return local.strftime("%d %b %Y · %H:%M"), local.strftime("%d %b · %H:%M")
    except (ValueError, TypeError):
        text = str(raw)
        return text, text


def _preview(message: str, n: int = 92) -> str:
    line = " ".join(str(message or "").split())
    return line if len(line) <= n else line[: n - 1] + "…"


def _unique(spans: list[str]) -> list[str]:
    seen: list[str] = []
    for raw in spans:
        s = " ".join(str(raw).split())
        if s and s not in seen:
            seen.append(s)
    return seen


def _ink_spans(spans: list[str]) -> list[str]:
    """Short slices only — do not yellow the whole letter."""
    return [s for s in _unique(spans) if 3 <= len(s) <= 40]


def _fallback_slice(text: str) -> str:
    """A short slice of their words, when no quote span lands on the letter."""
    text = " ".join(str(text or "").split())
    text = re.sub(r"^\s*urgent\b[\s,:-]*", "", text, flags=re.I).strip()
    if len(text) < 3:
        return ""
    if len(text) <= 40:
        return text
    cut = text[:40]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    cut = cut.strip(" ,;:-")
    return cut if len(cut) >= 3 else text[:40].strip()


def _ensure_spans(subject: str, message: str, spans: list[str]) -> list[str]:
    """Every letter gets the same yellow. One short slice, not the whole message."""
    ink = _ink_spans(spans)
    if "<mark>" in highlight(message, ink) or "<mark>" in highlight(subject, ink):
        return spans
    bit = _fallback_slice(message) or _fallback_slice(subject)
    return spans + [bit] if bit else spans


def _display_spans(spans: list[str], n: int = 4) -> list[str]:
    ranked = sorted(_unique(spans), key=lambda s: (0 if 12 <= len(s) <= 64 else 1, len(s)))
    out: list[str] = []
    for s in ranked:
        if any(s != t and t.casefold() in s.casefold() for t in out):
            continue
        out.append(s)
        if len(out) == n:
            break
    return out


def _card_quotes(spans: list[str], n: int = 2) -> list[str]:
    return _display_spans(spans, n)


def _score(raw: object) -> float | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _load_scores() -> pd.DataFrame:
    frames = []
    for path in SCORES.values():
        if path.exists():
            frames.append(pd.read_csv(path, usecols=["case_id", "model_score"]))
    if not frames:
        return pd.DataFrame(columns=["case_id", "model_score"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("case_id")


def _load_distil() -> pd.DataFrame:
    if not DISTIL_SCORES.exists():
        return pd.DataFrame(columns=["case_id", "distil_score", "distil_flagged"])
    raw = pd.read_csv(DISTIL_SCORES, usecols=["case_id", "model_score", "flagged"])
    return raw.rename(columns={"model_score": "distil_score", "flagged": "distil_flagged"})


def _draft_quote(subject: str, message: str) -> str:
    body = " ".join(str(message).split())
    sub = " ".join(str(subject).split()) or "your message"
    if len(body) < 40:
        raw = sub
    else:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", body) if p.strip()]
        raw = parts[0] if parts else sub
        if raw.casefold().startswith("urgent"):
            later = [p for p in parts[1:] if not p.casefold().startswith("urgent")]
            raw = later[-1] if later else sub
    if len(raw) > 88:
        raw = raw[:85].rsplit(" ", 1)[0].rstrip() + "…"
    return raw or "your message"


def _draft(subject: str, message: str, queue: str, send_allowed: int) -> str:
    """A draft the officer can read. It is not sent."""
    quote = _draft_quote(subject, message)
    who = "this unit" if queue == "General" else queue
    if not send_allowed:
        close = (
            f"{who} has this now.\n"
            "A person from the team will contact you. "
            "This desk will not send the reply in their place.\n"
            "Your message is not closed. It waits here until that person has read what you wrote."
        )
    else:
        close = (
            "This is not an urgent call.\n"
            "A person can write back when they choose. "
            "The desk will not send that note for them.\n"
            "The message stays in the inbox."
        )
    return f"Hello,\n\nYou wrote: “{quote}”\n\n{close}"


def _row(proc: pd.Series, exp: pd.Series) -> dict:
    subject = "" if pd.isna(proc.get("subject")) else str(proc["subject"])
    message = "" if pd.isna(proc.get("message")) else str(proc["message"])
    spans = _ensure_spans(subject, message, _parse_spans(exp.get("why_spans")))
    ink = _ink_spans(spans)
    desk_score = _score(proc.get("desk_score"))
    desk_flag_raw = proc.get("desk_flagged")
    flagged = 0 if desk_flag_raw is None or (isinstance(desk_flag_raw, float) and pd.isna(desk_flag_raw)) else int(desk_flag_raw)
    lexicon = int(exp.get("lexicon_hit", 0) or 0)
    vulnerable = int(exp.get("vulnerable", 0) or 0)
    send_allowed = 0 if flagged or lexicon or vulnerable else 1
    case_id = str(proc["case_id"])
    when, when_short = _when(proc["received_at"])
    out = {
        "case_id": case_id,
        "split": proc["split"],
        "channel": proc["channel"],
        "received_at": proc["received_at"],
        "when": when,
        "when_short": when_short,
        "subject": subject or "(no subject)",
        "message": message,
        "message_html": highlight(message, ink),
        "subject_html": highlight(subject or "(no subject)", ink),
        "preview": _preview(message),
        "flagged": flagged,
        "lexicon_hit": lexicon,
        "vulnerable": vulnerable,
        "queue_band": "human_now" if flagged else "normal",
        "suggested_queue": QUEUE_LABEL.get(str(exp.get("suggested_queue", "")), "General"),
        "queue_key": str(exp.get("suggested_queue", "general_complaints")),
        "spans": _display_spans(spans),
        "card_quotes": _card_quotes(spans),
        "ack_text": _draft(
            subject or "your message",
            message,
            QUEUE_LABEL.get(str(exp.get("suggested_queue", "")), "General"),
            send_allowed,
        ),
        "send_allowed": send_allowed,
        "send_lock_reason": "" if send_allowed else "Do not send a reply. A person must contact them.",
        "keyword_urgent": int(proc.get("mentions_urgent", 0) or 0),
        "score": desk_score,
        "score_label": f"{desk_score:.3f}" if desk_score is not None else "—",
        "cutoff": DISTIL_CUTOFF,
        "is_demo_sms": case_id == DEMO_SMS_P1,
        "is_demo_trap": case_id == DEMO_URGENT_TRAP,
        "is_known_miss": case_id == KNOWN_VAL_MISS,
    }
    out["path"] = path_copy(out)
    return out


def _take(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if frame.empty or n <= 0:
        return frame.iloc[0:0]
    return frame.sort_values("received_at", ascending=False).head(n)


def _pick_desk(merged: pd.DataFrame) -> pd.DataFrame:
    """A pitch desk, not the whole inbox.

    Keep the walkthrough cases, then add a mix: several channels,
    flagged harm, URGENT traps, and quiet mail that must stay visible.
    """
    must = merged[merged.case_id.isin(MUST_IDS)]
    pool = merged[~merged.case_id.isin(set(MUST_IDS))]
    channels = [c for c in ["SMS", "Email", "Web form", "Phone", "Facebook"] if c in set(pool.channel)]
    hot = pool.loc[pool.flagged == 1]
    cold = pool.loc[pool.flagged == 0]
    parts = [must]
    for channel in channels:
        parts.append(_take(hot.loc[hot.channel == channel], 4))
    parts.append(_take(hot, 8))
    for channel in channels:
        traps = cold.loc[(cold.channel == channel) & (cold.mentions_urgent == 1)]
        parts.append(_take(traps, 2))
    for channel in channels:
        quiet = cold.loc[(cold.channel == channel) & (cold.mentions_urgent == 0)]
        parts.append(_take(quiet, 3))
    return pd.concat(parts, ignore_index=True).drop_duplicates("case_id")


_INDEX: pd.DataFrame | None = None
_EXP: pd.DataFrame | None = None
_OPEN: dict[str, dict] = {}


def _index() -> pd.DataFrame:
    """Every message, for search. Not scored."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    cols = [
        "case_id",
        "split",
        "channel",
        "subject",
        "message",
        "received_at",
        "attachments",
        "is_after_hours",
        "mentions_tonight",
        "mentions_human",
        "mentions_urgent",
        "vulnerable",
    ]
    _INDEX = pd.read_csv(PROCESSED, usecols=cols)
    _INDEX["case_id"] = _INDEX["case_id"].astype(str)
    return _INDEX


def _explain_map() -> pd.DataFrame:
    global _EXP
    if _EXP is not None:
        return _EXP
    packs = [pd.read_csv(p) for p in EXPLAIN.values()]
    exp = pd.concat(packs, ignore_index=True).drop_duplicates("case_id")
    exp["case_id"] = exp["case_id"].astype(str)
    _EXP = exp.set_index("case_id")
    return _EXP


def search_messages(query: str, limit: int = 8) -> list[dict]:
    """Find by id or by words. Does not score, so the list stays fast."""
    q = " ".join((query or "").split())
    if len(q) < 2:
        return []
    idx = _index()
    ql = q.casefold()
    cid = idx["case_id"].str.casefold()
    exact = cid.eq(ql)
    prefix = cid.str.startswith(ql) & ~exact
    subject = idx["subject"].fillna("").str.casefold()
    message = idx["message"].fillna("").str.casefold()
    in_subject = subject.str.contains(ql, regex=False) & ~exact & ~prefix
    in_message = message.str.contains(ql, regex=False) & ~exact & ~prefix & ~in_subject
    chunks = [idx.loc[mask] for mask in (exact, prefix, in_subject, in_message) if mask.any()]
    if not chunks:
        return []
    found = pd.concat(chunks, ignore_index=True).drop_duplicates("case_id").head(limit)
    out = []
    for _, row in found.iterrows():
        text = "" if pd.isna(row["message"]) else " ".join(str(row["message"]).split())
        if len(text) > 90:
            text = text[:87].rsplit(" ", 1)[0] + "…"
        subject_text = "" if pd.isna(row["subject"]) else str(row["subject"]).strip()
        out.append(
            {
                "case_id": str(row["case_id"]),
                "channel": str(row["channel"]),
                "subject": subject_text or "(no subject)",
                "preview": text,
            }
        )
    return out


def _score_one(case_id: str) -> dict | None:
    idx = _index()
    hit = idx.loc[idx["case_id"] == str(case_id)]
    if hit.empty:
        return None
    frame = hit.iloc[[0]].copy()
    scored = predict_frame(frame)
    frame = frame.merge(scored, on="case_id", how="left")
    row = frame.iloc[0]
    exp_map = _explain_map()
    if case_id in exp_map.index:
        exp = exp_map.loc[case_id]
        if isinstance(exp, pd.DataFrame):
            exp = exp.iloc[0]
    else:
        exp = pd.Series(
            {
                "why_spans": "[]",
                "suggested_queue": "",
                "lexicon_hit": 0,
                "vulnerable": row.get("vulnerable", 0),
            }
        )
    return _row(row, exp)


_LOCK = threading.Lock()


def load_desk() -> list[dict]:
    """Deterministic pitch desk: flagged first, rest still listed."""
    if _CACHE is not None:
        return _CACHE
    with _LOCK:
        if _CACHE is not None:
            return _CACHE
        return _fill_desk()


def _fill_desk() -> list[dict]:
    global _CACHE
    missing = [p for p in [PROCESSED, *EXPLAIN.values()] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing " + ", ".join(str(p) for p in missing))

    proc = pd.read_csv(PROCESSED)
    packs = [pd.read_csv(p) for p in EXPLAIN.values()]
    exp = pd.concat(packs, ignore_index=True).drop_duplicates("case_id")
    merged = proc.merge(exp, on="case_id", how="inner", suffixes=("", "_exp"))
    merged = merged.merge(_load_scores(), on="case_id", how="left")

    desk = _pick_desk(merged)
    scored = predict_frame(desk)
    desk = desk.drop(columns=["desk_score", "desk_flagged"], errors="ignore").merge(
        scored, on="case_id", how="left"
    )

    items = [_row(r, r) for _, r in desk.iterrows()]
    hot = [i for i in items if i["queue_band"] == "human_now"]
    cold = [i for i in items if i["queue_band"] != "human_now"]
    hot.sort(key=lambda it: it["received_at"] or "", reverse=True)
    cold.sort(key=lambda it: it["received_at"] or "", reverse=True)
    sms = [i for i in hot if i["is_demo_sms"]]
    caught = [i for i in hot if i["is_known_miss"]]
    other_hot = [i for i in hot if not i["is_demo_sms"] and not i["is_known_miss"]]
    trap = [i for i in cold if i["is_demo_trap"]]
    rest = [i for i in cold if not i["is_demo_trap"]]
    _CACHE = sms + caught + other_hot + trap + rest
    return _CACHE


def get_item(case_id: str) -> dict | None:
    """Sample desk first. Any other id is scored on open, one message at a time."""
    for item in load_desk():
        if item["case_id"] == case_id:
            return item
    if case_id in _OPEN:
        return _OPEN[case_id]
    item = _score_one(case_id)
    if item is not None:
        _OPEN[case_id] = item
    return item


def counts() -> dict:
    items = load_desk()
    return {
        "n": len(items),
        "human_now": sum(1 for i in items if i["queue_band"] == "human_now"),
        "rest": sum(1 for i in items if i["queue_band"] != "human_now"),
        "generated": datetime.now().strftime("%d %b %Y"),
    }


def results() -> dict:
    """Locked numbers for the How page. The desk model is optimized DistilBERT."""
    return {
        "logistic_cut": f"{CUTOFF:.2f}",
        "distil_cut": f"{DISTIL_CUTOFF:.2f}",
        "val_logistic_recall": "0.992",
        "val_logistic_miss": "1 — MSG-028158 see attached (0.041)",
        "val_distil_recall": "1.00",
        "val_distil_miss": "0 — DistilBERT flags MSG-028158 (0.251)",
        "test_logistic_recall": "1.00",
        "test_logistic_flag": "2.9%",
        "test_dummy": "0",
        "test_keyword": "0",
        "redteam_logistic_recall": "1.00",
        "redteam_flag": "2.3%",
        "official": "The desk runs the optimized DistilBERT @0.25. That is the only joblib.",
    }
