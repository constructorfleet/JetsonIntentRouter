import json
from pathlib import Path

from flask import Blueprint, redirect, render_template, request

LABEL_UI = Blueprint("label_ui", __name__)

ROUTED_LOG = Path("logs/routed_clauses.jsonl")
OUT_FILE = Path("data/accepted_training.jsonl")
OUT_FILE.parent.mkdir(exist_ok=True)

INTENTS = ["MediaLibrary", "FetchMedia", "SearchNews", "CommandControl", "Unknown"]


def load_unreviewed(limit=50):
    rows = []
    if not ROUTED_LOG.exists():
        return rows

    with ROUTED_LOG.open() as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


@LABEL_UI.route("/label", methods=["GET", "POST"])
def label():
    if request.method == "POST":
        text = request.form["text"]
        label = request.form["label"]

        with OUT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"text": text, "label": label}) + "\n")

        return redirect("/label")

    data = load_unreviewed()
    return render_template(
        "label.html",
        data=data,
        intents=INTENTS,
    )
