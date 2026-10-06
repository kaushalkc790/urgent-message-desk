"""Offline pack: ten scored messages, no model weights.

The officer desk pushes up to ten messages here. The zip is rebuilt from
that list. Five open on the page. See more shows the rest.
The full DistilBERT file stays in models/ and is not copied.
"""

from __future__ import annotations

import csv
import html
import io
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DESK = ROOT / "outputs" / "desk"
PACK = ROOT / "outputs" / "packs"
TRAY = DESK / "offline_tray.json"
OUT_HTML = PACK / "offline_desk.html"
ZIP = PACK / "offline_desk.zip"
STAGE = PACK / "_offline_desk"

CAP = 10
SAMPLE_N = 5
DARWIN = ZoneInfo("Australia/Darwin")


def _now() -> str:
    return datetime.now(DARWIN).strftime("%d %b %Y · %H:%M")


def _read() -> dict:
    if not TRAY.exists():
        return {}
    try:
        data = json.loads(TRAY.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_tray(items: list[dict], built_at: str) -> None:
    DESK.mkdir(parents=True, exist_ok=True)
    TRAY.write_text(
        json.dumps({"built_at": built_at, "items": items}, indent=2),
        encoding="utf-8",
    )


def items() -> list[dict]:
    raw = _read().get("items") or []
    return [row for row in raw if isinstance(row, dict) and row.get("case_id")]


def status() -> dict:
    rows = items()
    built = str(_read().get("built_at") or "")
    return {
        "items": rows,
        "ids": [row["case_id"] for row in rows],
        "n": len(rows),
        "cap": CAP,
        "full": len(rows) >= CAP,
        "built_at": built,
        "sample_n": min(SAMPLE_N, len(rows)),
        "more_n": max(0, len(rows) - SAMPLE_N),
    }


def snapshot(item: dict) -> dict:
    score = item.get("score")
    try:
        pct = round(float(score) * 100, 1)
    except (TypeError, ValueError):
        pct = 0.0
    flagged = int(item.get("flagged") or 0)
    if item.get("is_known_miss") and flagged:
        tag = "Urgent. DistilBERT flags the row logistic missed."
    elif flagged:
        tag = "Urgent. Escalate. Do not send a reply."
    elif item.get("keyword_urgent"):
        tag = "Not urgent. The word URGENT is not a reason."
    else:
        tag = "Not urgent. Do not escalate. It stays listed."
    path = item.get("path") or {}
    quote = str(path.get("quote") or "")
    if not quote and item.get("card_quotes"):
        quote = str(item["card_quotes"][0])
    if not quote:
        quote = str(item.get("message") or "")
    return {
        "case_id": str(item["case_id"]),
        "channel": str(item.get("channel") or ""),
        "subject": str(item.get("subject") or "(no subject)"),
        "message": str(item.get("message") or ""),
        "quote": quote,
        "queue": str(item.get("suggested_queue") or "General"),
        "when": str(item.get("when") or ""),
        "pct_sure": pct,
        "flagged": flagged,
        "tag": tag,
    }


def _choose_default(desk: list[dict]) -> list[dict]:
    from app.data import DEMO_SMS_P1, DEMO_URGENT_TRAP, KNOWN_VAL_MISS

    by_id = {it["case_id"]: it for it in desk}
    head = [by_id[cid] for cid in (DEMO_SMS_P1, DEMO_URGENT_TRAP, KNOWN_VAL_MISS) if cid in by_id]
    used = {it["case_id"] for it in head}
    rest = [it for it in desk if it["case_id"] not in used]
    hot = [it for it in rest if it.get("flagged")]
    cold = [it for it in rest if not it.get("flagged")]
    extra = hot[:4] + cold[:3]
    if len(extra) < 7:
        seen = {it["case_id"] for it in extra}
        for it in rest:
            if it["case_id"] in seen:
                continue
            extra.append(it)
            if len(extra) >= 7:
                break
    return [snapshot(it) for it in (head + extra)[:CAP]]


def ensure_seed(desk: list[dict]) -> None:
    """First run only: the walkthrough messages, plus more from the desk, up to ten."""
    if TRAY.exists():
        return
    chosen = _choose_default(desk)
    if not chosen:
        return
    _write_tray(chosen, "")
    write_pack(chosen)


def add(item: dict) -> str:
    rows = items()
    case_id = str(item["case_id"])
    if any(row["case_id"] == case_id for row in rows):
        return f"{case_id} is already in the offline pack."
    if len(rows) >= CAP:
        return "The pack holds 10. Remove one, or clear it, then add this message."
    rows.append(snapshot(item))
    built = write_pack(rows)
    place = "on the page" if len(rows) <= SAMPLE_N else "under See more"
    return f"Added {case_id} {place}. The pack has {len(rows)} of 10. Download the updated zip ({built})."


def remove(case_id: str) -> str:
    rows = items()
    idx = next((i for i, row in enumerate(rows) if row["case_id"] == case_id), -1)
    if idx < 0:
        return f"{case_id} was not in the pack. Nothing was deleted from the inbox."
    was_sample = idx < SAMPLE_N
    rows.pop(idx)
    built = write_pack(rows)
    moved = " The next message moved onto the page." if was_sample and rows else ""
    return (
        f"Removed {case_id} from the pack. The inbox still has it. "
        f"The zip was rebuilt at {built}. {len(rows)} of 10 are in it.{moved}"
    )


def feature(case_id: str) -> str:
    rows = items()
    idx = next((i for i, row in enumerate(rows) if row["case_id"] == case_id), -1)
    if idx < 0:
        return f"{case_id} is not in the pack."
    if idx < SAMPLE_N:
        return f"{case_id} is already one of the five on the page."
    row = rows.pop(idx)
    rows.insert(0, row)
    built = write_pack(rows)
    return f"{case_id} is now one of the five on the page. The zip was rebuilt ({built})."


def clear() -> str:
    write_pack([])
    return "The pack is empty. Add up to 10 messages, then download. Nothing in the inbox was deleted."


def zip_path() -> Path | None:
    rows = items()
    if not rows:
        return None
    if not ZIP.exists():
        write_pack(rows)
    return ZIP


def write_pack(rows: list[dict]) -> str:
    """Rebuild the small zip from these rows. Does not copy model weights."""
    built = _now()
    _write_tray(rows, built)
    if STAGE.exists():
        shutil.rmtree(STAGE)
    samples = STAGE / "samples"
    samples.mkdir(parents=True)
    page = render_page(rows, built)
    (STAGE / "offline_desk.html").write_text(page, encoding="utf-8")
    (STAGE / "README_OFFLINE.txt").write_text(_readme(rows, built), encoding="utf-8")
    (STAGE / "score_samples.py").write_text(_score_script(), encoding="utf-8")
    (samples / "messages.csv").write_text(_csv(rows), encoding="utf-8")
    _dump_joblib(rows, samples / "distilbert_samples.joblib")

    PACK.mkdir(parents=True, exist_ok=True)
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in STAGE.rglob("*"):
            if path.is_file():
                zf.write(path, Path("offline_desk") / path.relative_to(STAGE))
    shutil.copy2(STAGE / "offline_desk.html", OUT_HTML)
    shutil.rmtree(STAGE)
    return built


def render_page(rows: list[dict], built_at: str) -> str:
    rows = _paint(rows)
    shown = rows[:SAMPLE_N]
    extra = rows[SAMPLE_N:]
    shown_html = "\n".join(_card(row) for row in shown) or "<p>No messages in this pack yet.</p>"
    more_html = ""
    if extra:
        more_html = (
            '<details class="see-more"><summary>See more</summary>'
            + "\n".join(_card(row) for row in extra)
            + "</details>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Correspondence triage — offline pack</title>
  <style>
    :root {{ --ink:#1a1814; --muted:#5b5348; --sand:#f3eee6; --paper:#fffcf7; --line:#d9d0c2; --alert:#9a1b1b; --ok:#2c5a3a; --steel:#3a536b; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: var(--ink); background: var(--sand); line-height: 1.45; }}
    header, main, footer {{ max-width: 880px; margin: 0 auto; padding: 1.1rem 1.25rem; }}
    header {{ background: var(--paper); border-bottom: 1px solid var(--line); max-width: none; }}
    header div {{ max-width: 880px; margin: 0 auto; }}
    .kicker {{ margin: 0; letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.72rem; color: var(--muted); }}
    h1 {{ margin: 0.2rem 0 0; font-family: Palatino, "Palatino Linotype", serif; font-size: 1.7rem; font-weight: 500; }}
    .tag {{ margin: 0.2rem 0 0; color: var(--muted); }}
    h2 {{ font-family: Palatino, "Palatino Linotype", serif; font-size: 1.25rem; font-weight: 500; margin: 1.4rem 0 0.4rem; }}
    ul.rules {{ list-style: none; padding: 0; margin: 0.4rem 0 0; }}
    ul.rules li {{ margin: 0.35rem 0; }}
    ul.rules strong {{ color: var(--alert); }}
    .card {{ background: var(--paper); border: 1px solid var(--line); border-top: 3px solid var(--alert); padding: 0.85rem 1rem; margin: 0.7rem 0; }}
    .card.quiet {{ border-top-color: var(--steel); }}
    .meta {{ margin: 0; font-size: 0.78rem; color: var(--muted); }}
    .card h3 {{ margin: 0.25rem 0; font-size: 1.05rem; }}
    blockquote {{ margin: 0.4rem 0; font-family: Palatino, "Palatino Linotype", serif; }}
    mark {{ background: #f3de8a; color: inherit; padding: 0 0.08em; }}
    .sure {{ margin: 0.55rem 0 0; font-weight: 700; }}
    .sure span {{ font-weight: 500; color: var(--muted); }}
    .meter {{ position: relative; height: 10px; background: #e7e0d4; margin-top: 0.35rem; }}
    .meter i {{ display: block; height: 100%; background: var(--alert); }}
    .card.quiet .meter i {{ background: var(--steel); }}
    .meter b {{ position: absolute; top: -3px; bottom: -3px; left: 25%; width: 2px; background: var(--ink); }}
    .scale {{ display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--muted); margin-top: 0.2rem; }}
    details.see-more {{ background: transparent; border: 0; margin: 0.1rem 0 1rem; }}
    details.see-more > summary {{
      width: fit-content; cursor: pointer; padding: 0; font-size: 0.78rem;
      color: var(--steel); list-style: none;
    }}
    details.see-more > summary::-webkit-details-marker {{ display: none; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--paper); font-size: 0.92rem; }}
    th, td {{ text-align: left; padding: 0.45rem 0.5rem; border-bottom: 1px solid var(--line); }}
    th {{ font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }}
    tr.desk td {{ font-weight: 700; }}
    footer {{ color: var(--muted); font-size: 0.8rem; }}
    code {{ font-family: Consolas, monospace; font-size: 0.88em; }}
  </style>
</head>
<body>
  <header>
    <div>
      <p class="kicker">Northern Territory Government · Public contact unit</p>
      <h1>Correspondence triage</h1>
      <p class="tag">Offline pack of {len(rows)}. The model suggests. You decide. No internet needed.</p>
      <p class="tag">Built {html.escape(built_at)}. These scores come from the officer desk. This file does not contain the model weights.</p>
    </div>
  </header>
  <main>
    <h2>The four rules</h2>
    <ul class="rules">
      <li><strong>You decide.</strong> The model only suggests.</li>
      <li><strong>A miss costs more</strong> than an extra alarm.</li>
      <li><strong>One cutoff, 25%,</strong> for every message. Same line for every channel and community.</li>
      <li><strong>Never delete</strong> a message the model did not flag.</li>
    </ul>

    {shown_html}
    {more_html}

    <h2>What the buttons mean</h2>
    <ul class="rules">
      <li><strong>Escalate</strong> — a person from the unit contacts them. Do not send a reply.</li>
      <li><strong>Not urgent</strong> — you say it is not urgent. It stays in the inbox. Nothing is deleted.</li>
    </ul>

    <h2>Why this model</h2>
    <table>
      <thead><tr><th>Check</th><th>What it finds</th></tr></thead>
      <tbody>
        <tr><td>Dummy</td><td>Finds 0 urgent messages</td></tr>
        <tr><td>The word URGENT</td><td>A trap. Test recall 0. Finds none of the urgent messages.</td></tr>
        <tr><td>Earlier logistic at 0.38</td><td>Missed “see attached”</td></tr>
        <tr class="desk"><td>Optimized DistilBERT at 25%</td><td>Validation recall 1.00. Miss 0. Extra 0. This runs the desk.</td></tr>
      </tbody>
    </table>
    <p>These {len(rows)} scores are saved in <code>samples/distilbert_samples.joblib</code>. The full weight file is about 256 MB, so it is not copied into this pack.</p>
  </main>
  <footer>
    <p>Unflagged mail stays visible. The model does not send a reply. The model does not close a message.</p>
  </footer>
</body>
</html>
"""


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _paint(rows: list[dict]) -> list[dict]:
    """Same yellow slices as the desk. A letter with no span still gets one."""
    import pandas as pd

    from app.data import _ensure_spans, _explain_map, _ink_spans, _parse_spans, highlight

    exp_map = _explain_map()
    painted = []
    for row in rows:
        row = dict(row)
        subject = str(row.get("subject") or "")
        message = str(row.get("message") or "")
        spans: list[str] = []
        case_id = str(row.get("case_id") or "")
        if case_id in exp_map.index:
            exp = exp_map.loc[case_id]
            if isinstance(exp, pd.DataFrame):
                exp = exp.iloc[0]
            spans = _parse_spans(exp.get("why_spans"))
        spans = _ensure_spans(subject, message, spans)
        ink = _ink_spans(spans)
        row["subject_html"] = highlight(subject or "(no subject)", ink)
        row["message_html"] = highlight(message, ink)
        painted.append(row)
    return painted


def _card(row: dict) -> str:
    quiet = "" if row.get("flagged") else " quiet"
    pct = float(row.get("pct_sure") or 0)
    width = min(max(pct, 0.0), 100.0)
    body = row.get("message_html") or _esc(row.get("message") or row.get("quote") or "")
    title = row.get("subject_html") or _esc(row.get("subject"))
    return f"""<article class="card{quiet}" id="{_esc(row.get("case_id"))}">
      <p class="meta">{_esc(row.get("channel"))} · {_esc(row.get("case_id"))} · {_esc(row.get("queue"))}</p>
      <h3>{title}</h3>
      <blockquote>{body}</blockquote>
      <p class="sure">{pct:.1f}% sure <span>{_esc(row.get("tag"))}</span></p>
      <div class="meter" aria-hidden="true"><i style="width:{width:.1f}%"></i><b></b></div>
      <div class="scale"><span>0%</span><span>25% — escalate from here</span><span>100%</span></div>
    </article>"""


def _csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    fields = ["case_id", "channel", "subject", "message", "pct_sure", "flagged", "queue", "tag"]
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def _dump_joblib(rows: list[dict], path: Path) -> None:
    import joblib

    bundle = {
        "name": "distilbert-optimized-samples",
        "source": "models/distilbert_optimized.joblib",
        "cutoff": 0.25,
        "samples": [
            {
                "case_id": row["case_id"],
                "channel": row["channel"],
                "subject": row["subject"],
                "pct_sure": row["pct_sure"],
                "flagged": row["flagged"],
                "tag": row["tag"],
            }
            for row in rows
        ],
    }
    joblib.dump(bundle, path)


def _readme(rows: list[dict], built_at: str) -> str:
    return f"""Offline pack — {len(rows)} messages

Open offline_desk.html in a browser. No internet. No download.

Five messages are on the page. See more shows the rest. Nothing is deleted.

samples/distilbert_samples.joblib holds these scores from optimized
DistilBERT at 25%. It is small.

The full weights stay in the project at models/distilbert_optimized.joblib
(about 256 MB). They are not in this zip.

Built {built_at} from the officer desk. To take a different ten, add them
on the desk and download the zip again.

    python score_samples.py
"""


def _score_script() -> str:
    return '''"""Print the packed scores. No download. No model weights."""
from pathlib import Path

import joblib

bundle = joblib.load(Path(__file__).resolve().parent / "samples" / "distilbert_samples.joblib")
print(bundle["name"], "cutoff", bundle["cutoff"])
rows = bundle["samples"]
print(f"{len(rows)} messages. Five are on the page. See more shows the rest.")
for i, row in enumerate(rows, start=1):
    place = "on the page" if i <= 5 else "under See more"
    print(f"{i}. {row['case_id']}  {row['pct_sure']}%  {place}  {row['tag']}")
'''
