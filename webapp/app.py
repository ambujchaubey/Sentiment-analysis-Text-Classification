"""Flask web app: live sentiment + emotion analysis, batch CSV analysis and review insights dashboard.

    python webapp/build_artifacts.py   # once
    python webapp/app.py               # then open http://127.0.0.1:5000
"""
import io
import json
import os
from collections import Counter

import pandas as pd
from flask import Flask, jsonify, render_template, request

import nlp

MAX_TEXT_LEN = 5000
MAX_BATCH_ROWS = 5000
MAX_AI_BATCH_ROWS = 1000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.json.ensure_ascii = False


def _load_json(name):
    with open(os.path.join(nlp.ARTIFACTS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


METRICS = _load_json("metrics.json")
INSIGHTS = _load_json("insights.json")
REVIEWS = pd.read_csv(os.path.join(nlp.ARTIFACTS_DIR, "reviews_scored.csv"))
REVIEWS["replyContent"] = REVIEWS["replyContent"].fillna("")
nlp.load_model()
nlp.transformers_available()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    return jsonify(
        models="transformer" if nlp.transformers_available() else "classic",
        emotion_model=nlp.EMOTION_MODEL_ID,
        sentiment_model=nlp.SENTIMENT_MODEL_ID,
        max_ai_batch_rows=MAX_AI_BATCH_ROWS,
        max_batch_rows=MAX_BATCH_ROWS,
    )


@app.post("/api/analyze")
def api_analyze():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="Please enter some text."), 400
    return jsonify(nlp.analyze(text[:MAX_TEXT_LEN]))


def _pick_text_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in df.columns:
        return requested
    for name in ["text", "content", "review", "comment", "message", "feedback", "tweet", "body"]:
        for col in df.columns:
            if col.strip().lower() == name:
                return col
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        raise ValueError("No text column found in the CSV.")
    return max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())


@app.post("/api/batch")
def api_batch():
    try:
        if "file" in request.files:
            df = pd.read_csv(io.BytesIO(request.files["file"].read()), encoding_errors="replace")
            col = _pick_text_column(df, request.form.get("column"))
            texts = df[col].dropna().astype(str).tolist()
            fast = request.form.get("fast") == "1"
        else:
            payload = request.get_json(silent=True) or {}
            texts, col = [t for t in payload.get("texts", []) if str(t).strip()], "text"
            fast = bool(payload.get("fast"))
    except Exception as exc:  # malformed CSV etc.
        return jsonify(error=f"Could not read input: {exc}"), 400

    if not texts:
        return jsonify(error="No text rows found."), 400
    fast = fast or not nlp.transformers_available()
    limit = MAX_BATCH_ROWS if fast else MAX_AI_BATCH_ROWS
    truncated = len(texts) > limit
    texts = [t[:MAX_TEXT_LEN] for t in texts[:limit]]

    rows = []
    for r in nlp.analyze_many(texts, fast=fast):
        s = r["sentiment"]
        rows.append({
            "text": r["text"],
            "sentiment": s["label"],
            "score": round(s["roberta"]["polarity"] if "roberta" in s else s["vader"]["compound"], 4),
            "vader": s["vader"]["label"],
            "compound": s["vader"]["compound"],
            "afinn": s["afinn"]["score"],
            "emotion": r["emotion"]["label"],
            "confidence": round(r["emotion"]["confidence"], 4),
        })

    labels = nlp.EMOTIONS if fast else nlp.AI_EMOTIONS
    sent = Counter(r["sentiment"] for r in rows)
    emo = Counter(r["emotion"] for r in rows)
    return jsonify(
        column=col,
        truncated=truncated,
        limit=limit,
        models="classic" if fast else "transformer",
        summary={
            "count": len(rows),
            "avg_score": round(sum(r["score"] for r in rows) / len(rows), 4),
            "sentiment": {k: sent.get(k, 0) for k in ["positive", "neutral", "negative"]},
            "emotion": {k: emo.get(k, 0) for k in labels},
        },
        rows=rows,
    )


@app.get("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.get("/api/insights")
def api_insights():
    return jsonify(INSIGHTS)


@app.get("/api/reviews")
def api_reviews():
    df = REVIEWS
    for field in ["app", "sentiment", "emotion"]:
        value = request.args.get(field)
        if value:
            df = df[df[field] == value]
    q = request.args.get("q", "").strip().lower()
    if q:
        df = df[df["content"].str.contains(q, case=False, regex=False, na=False)]

    sort = request.args.get("sort", "helpful")
    if sort == "most_negative":
        df = df.sort_values("score")
    elif sort == "most_positive":
        df = df.sort_values("score", ascending=False)
    else:
        df = df.sort_values("thumbsUpCount", ascending=False)

    page = max(int(request.args.get("page", 1)), 1)
    size = min(max(int(request.args.get("size", 20)), 1), 100)
    chunk = df.iloc[(page - 1) * size: page * size]
    return jsonify(total=int(len(df)), page=page, size=size,
                   items=chunk.fillna("").to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=int(os.environ.get("PORT", 5000)))
