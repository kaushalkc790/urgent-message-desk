"""Copy for the four-step path. Desk model is the optimized DistilBERT."""

from __future__ import annotations


def path_copy(item: dict) -> dict:
    flagged = bool(item.get("flagged"))
    send_off = not bool(item.get("send_allowed"))
    quotes = item.get("card_quotes") or []
    quote = quotes[0] if quotes else "their wording"
    if item.get("is_known_miss") and flagged:
        error = (
            "Guess from distilbert_optimized.joblib. "
            "Epoch 1, cutoff 0.25. This is the row logistic missed. "
            "Val: recall 1.00, miss 0, extra 0. Accuracy is the wrong bar."
        )
    elif flagged:
        error = (
            "Guess from the optimized DistilBERT, not a fact. "
            "AdamW, class-weighted loss, cutoff 0.25. "
            "Val: recall 1.00, miss 0, extra 0. "
            "A dummy that never flags is ~97% accurate and finds 0 P1s."
        )
    else:
        error = (
            "Guess from the optimized DistilBERT, not a fact. "
            "Score is under 0.25, so it stays listed. "
            "Val: recall 1.00, miss 0."
        )
    return {
        "quote": quote,
        "guess": item.get("score_label") or "—",
        "flag_word": "Flag" if flagged else "No flag",
        "compare": "≥ 0.25" if flagged else "< 0.25",
        "band": "Human now" if flagged else "Rest of inbox",
        "action": "Escalate" if flagged else "Still listed",
        "action_note": "Do not send a reply" if send_off else "A person can send this",
        "error": error,
        "file": "distilbert_optimized.joblib",
    }
