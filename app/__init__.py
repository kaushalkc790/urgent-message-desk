"""Correspondence triage — Flask desk.

Hard rules (keep these in the UI and in this file):
- The model assists. A human decides.
- Miss costs more than an extra alarm.
- One cutoff (0.25) for every style and community.
- Never delete or hide unflagged mail.
- Never auto-reply. Do not send a reply on serious items.
- Reasons quote the writer. Do not polish their English.
- community is never a score.
"""

from __future__ import annotations

import threading
from pathlib import Path

from flask import Flask, abort, jsonify, make_response, redirect, render_template, request, send_file, send_from_directory, session, url_for

from app.data import (
    DEMO_SMS_P1,
    DEMO_URGENT_TRAP,
    KNOWN_VAL_MISS,
    get_item,
    load_desk,
    results,
    search_messages,
)
from app import decisions, offline


_LOADING = """<!doctype html>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>Opening the desk</title>
<body style="font-family: Georgia, serif; background:#f4ede1; color:#1f1a14; margin:3rem;">
<p>The desk is opening the model. This page refreshes itself.</p>
</body>
"""
_READY = threading.Event()


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = "nt-public-contact-pitch-desk"
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    def _warm() -> None:
        try:
            load_desk()
        except Exception:
            app.logger.exception("The desk model did not open")
        finally:
            _READY.set()

    threading.Thread(target=_warm, name="desk-warm", daemon=True).start()

    @app.before_request
    def wait_for_model():
        if request.path.startswith("/static") or _READY.is_set():
            return None
        return make_response(_LOADING)

    @app.context_processor
    def inject_desk() -> dict:
        desk = load_desk()
        offline.ensure_seed(desk)
        pack = offline.status()
        items = _with_decisions(desk)
        return {
            "counts": _tallies(items),
            "saved": decisions.recent(6),
            "demo_sms": DEMO_SMS_P1,
            "demo_trap": DEMO_URGENT_TRAP,
            "demo_miss": KNOWN_VAL_MISS,
            "cutoff": "0.25",
            "distil_cutoff": "0.25",
            "not_found": False,
            "pack": pack,
            "pack_ids": set(pack["ids"]),
        }

    @app.get("/how")
    def how():
        return render_template("how.html", results=results())

    @app.get("/charts/<name>")
    def charts(name: str):
        allowed = {"05_decision_flow.png", "06_technical_flow.png"}
        if name not in allowed:
            abort(404)
        folder = Path(__file__).resolve().parents[1] / "outputs" / "charts"
        return send_from_directory(folder, name)

    @app.get("/suggest")
    def suggest():
        return jsonify(search_messages(request.args.get("q", "")))

    @app.get("/")
    def inbox():
        items = _with_decisions(load_desk())
        query = (request.args.get("q") or "").strip()
        hits = search_messages(query) if query else []
        open_id = request.args.get("open")
        if not open_id and len(hits) == 1 and hits[0]["case_id"].casefold() == query.casefold():
            open_id = hits[0]["case_id"]
        if not open_id:
            open_id = _default_open(items)
        item = get_item(open_id) if open_id else None
        if open_id and item is None:
            abort(404)
        if item:
            item = _apply_decision(item)
        return render_template(
            "inbox.html",
            items=items,
            item=item,
            query=query,
            hits=hits,
            notice=session.pop("notice", None),
        )

    @app.post("/decide/<case_id>")
    def decide(case_id: str):
        item = get_item(case_id)
        if item is None:
            abort(404)
        action = request.form.get("action", "")
        if action == "escalate":
            saved = decisions.record(case_id, "escalated", item)
            session["notice"] = (
                f"Saved in outputs/desk/escalated.json at {saved['at']}. Escalated. A person from the unit will contact them. Do not send a reply."
            )
        elif action == "not_urgent":
            saved = decisions.record(case_id, "not_urgent", item)
            session["notice"] = (
                f"Saved in outputs/desk/not_urgent.json at {saved['at']}. You marked this not urgent. It moved to the rest of the inbox. You can still open it. Nothing was deleted."
            )
        elif action == "send":
            prior = decisions.get(case_id) or {}
            if item["send_allowed"] and prior.get("action") != "escalated":
                session["notice"] = (
                    "A person still has to send this. The model will not send it."
                )
            else:
                session["notice"] = "Do not send a reply. A person must contact them."
        return redirect(url_for("inbox", open=case_id))

    @app.post("/offline/add/<case_id>")
    def offline_add(case_id: str):
        item = get_item(case_id)
        if item is None:
            abort(404)
        session["notice"] = offline.add(item)
        return _back(case_id)

    @app.post("/offline/remove/<case_id>")
    def offline_remove(case_id: str):
        session["notice"] = offline.remove(case_id)
        return _back(request.form.get("open") or case_id)

    @app.post("/offline/feature/<case_id>")
    def offline_feature(case_id: str):
        session["notice"] = offline.feature(case_id)
        return _back(request.form.get("open") or case_id)

    @app.post("/offline/clear")
    def offline_clear():
        session["notice"] = offline.clear()
        return _back(request.form.get("open"))

    @app.get("/offline/download")
    def offline_download():
        path = offline.zip_path()
        if path is None:
            session["notice"] = "The pack is empty. Add up to 10 messages, then download."
            return _back(request.args.get("open"))
        return send_file(path, as_attachment=True, download_name="offline_desk.zip")

    @app.errorhandler(404)
    def missing(_err):
        return (
            render_template(
                "inbox.html",
                items=_with_decisions(load_desk()),
                item=None,
                notice="No message with that id. Nothing was deleted.",
                not_found=True,
            ),
            404,
        )

    return app


def _back(open_id: str | None):
    if open_id:
        return redirect(url_for("inbox", open=open_id))
    return redirect(url_for("inbox"))


def _default_open(items: list[dict]) -> str:
    for it in items:
        if it["case_id"] == DEMO_SMS_P1:
            return DEMO_SMS_P1
    return items[0]["case_id"] if items else ""


def _apply_decision(item: dict) -> dict:
    out = dict(item)
    saved = decisions.get(item["case_id"]) or {}
    out["decision"] = saved.get("action") or ""
    out["saved_at"] = saved.get("at") or ""
    if out["decision"] == "not_urgent":
        out["queue_band"] = "normal"
    return out


def _tallies(items: list[dict]) -> dict:
    return {
        "n": len(items),
        "human_now": sum(1 for i in items if i["queue_band"] == "human_now"),
        "rest": sum(1 for i in items if i["queue_band"] != "human_now"),
        "escalated": sum(1 for i in items if i.get("decision") == "escalated"),
        "not_urgent": sum(1 for i in items if i.get("decision") == "not_urgent"),
    }


def _with_decisions(items: list[dict]) -> list[dict]:
    applied = [_apply_decision(i) for i in items]
    hot = [i for i in applied if i["queue_band"] == "human_now"]
    cold = [i for i in applied if i["queue_band"] != "human_now"]
    sms = [i for i in hot if i.get("is_demo_sms")]
    other_hot = [i for i in hot if not i.get("is_demo_sms")]
    trap = [i for i in cold if i.get("is_demo_trap")]
    miss = [i for i in cold if i.get("is_known_miss")]
    rest = [i for i in cold if not i.get("is_demo_trap") and not i.get("is_known_miss")]
    return sms + other_hot + trap + miss + rest


app = create_app()
