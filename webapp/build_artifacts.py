"""Train / evaluate the models and pre-compute review insights for the web dashboard.

    python webapp/build_artifacts.py

The first run scores all reviews with the transformer models (~45 min on CPU). Scores are
checkpointed to artifacts/review_scores.csv, so an interrupted run resumes where it stopped
and later runs take about a minute. Set SENTISCOPE_FAST=1 to skip the transformer models.
"""
import json
import os
import sys
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nlp  # noqa: E402

EMOTIONS_CSV = os.path.join(nlp.ROOT_DIR, "Emotions_training.csv")
REVIEWS_CSV = os.path.join(nlp.ROOT_DIR, "Dataprocessed_data.csv")
SPOTCHECK_CSV = os.path.join(nlp.BASE_DIR, "data", "review_spotcheck.csv")
METRICS_PATH = os.path.join(nlp.ARTIFACTS_DIR, "metrics.json")
INSIGHTS_PATH = os.path.join(nlp.ARTIFACTS_DIR, "insights.json")
SCORED_REVIEWS_PATH = os.path.join(nlp.ARTIFACTS_DIR, "reviews_scored.csv")
REVIEW_SCORES_CACHE = os.path.join(nlp.ARTIFACTS_DIR, "review_scores.csv")
SCORE_CHUNK = 512

SHARED_EMOTIONS = ["sadness", "joy", "anger", "fear", "surprise"]

APP_NAMES = {
    "com.anydo": "Any.do",
    "com.todoist": "Todoist",
    "com.microsoft.todos": "Microsoft To Do",
    "com.tasks.android": "Tasks",
    "com.ticktick.task": "TickTick",
    "com.habitrpg.android.habitica": "Habitica",
    "com.oristats.habitbull": "HabitBull",
    "prox.lab.calclock": "CalClock",
    "com.gmail.jmartindev.timetune": "TimeTune",
    "cc.forestapp": "Forest",
    "com.habitnow": "HabitNow",
    "com.appgenix.bizcal": "Business Calendar",
    "com.appxy.planner": "Planner Pro",
    "com.levor.liferpgtasks": "LifeRPG",
    "com.artfulagenda.app": "Artful Agenda",
}


def train_classic_model():
    print("Training classic emotion model ...")
    df = pd.read_csv(EMOTIONS_CSV)
    df["clean"] = df["text"].map(nlp.clean_text)

    train_val, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train, val = train_test_split(train_val, test_size=0.125, random_state=42, stratify=train_val["label"])

    model = make_pipeline(
        TfidfVectorizer(sublinear_tf=True),
        LogisticRegression(C=10, max_iter=3000, class_weight="balanced", random_state=42),
    )
    model.fit(train["clean"], train["label"])

    accs = {name: float(accuracy_score(part["label"], model.predict(part["clean"])))
            for name, part in [("train", train), ("validation", val), ("test", test)]}
    y_pred = model.predict(test["clean"])
    report = classification_report(test["label"], y_pred, target_names=nlp.EMOTIONS, output_dict=True)

    vec, clf = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
    vocab = np.array(vec.get_feature_names_out())
    top_words = {nlp.EMOTIONS[int(c)]: vocab[np.argsort(clf.coef_[i])[-12:][::-1]].tolist()
                 for i, c in enumerate(clf.classes_)}

    os.makedirs(nlp.ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, nlp.MODEL_PATH)
    nlp.load_model.cache_clear()
    print("  accuracy:", {k: round(v, 4) for k, v in accs.items()})
    return test, {
        "model": "TF-IDF + Logistic Regression (C=10, balanced)",
        "baseline_notebook": {"train": 0.9309, "validation": 0.8231, "test": 0.8309},
        "accuracy": accs,
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "report": {k: report[k] for k in nlp.EMOTIONS},
        "macro_f1": report["macro avg"]["f1-score"],
        "confusion_matrix": confusion_matrix(test["label"], y_pred).tolist(),
        "labels": nlp.EMOTIONS,
        "distribution": {nlp.EMOTIONS[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "top_words": top_words,
    }


def compare_models(test_df):
    """Classic vs transformer on (a) review-style spot-check set, (b) original test set."""
    print("Comparing classic and transformer models ...")
    spot = pd.read_csv(SPOTCHECK_CSV)
    results = nlp.analyze_many(spot["text"].tolist())
    spot["ai_emotion"] = [r["emotion"]["label"] for r in results]
    spot["classic_emotion"] = [r["emotion_classic"]["label"] for r in results]
    spot["roberta"] = [r["sentiment"]["label"] for r in results]
    spot["vader"] = [r["sentiment"]["vader"]["label"] for r in results]
    spot["afinn"] = [r["sentiment"]["afinn"]["label"] for r in results]

    emo_rows = spot[spot["emotion"].isin(SHARED_EMOTIONS)]
    spotcheck = {
        "size": int(len(spot)),
        "emotion_size": int(len(emo_rows)),
        "emotion_accuracy": {
            "transformer": float((emo_rows["ai_emotion"] == emo_rows["emotion"]).mean()),
            "classic": float((emo_rows["classic_emotion"] == emo_rows["emotion"]).mean()),
        },
        "sentiment_accuracy": {
            name: float((spot[name] == spot["sentiment"]).mean()) for name in ["roberta", "vader", "afinn"]
        },
        "examples": spot[["text", "emotion", "ai_emotion", "classic_emotion", "sentiment", "roberta", "vader"]]
            .to_dict(orient="records"),
    }

    shared = test_df[test_df["label"].map(lambda l: nlp.EMOTIONS[l]).isin(SHARED_EMOTIONS)]
    truth = shared["label"].map(lambda l: nlp.EMOTIONS[l]).to_numpy()
    emo_probs, _ = nlp.transformer_scores(shared["text"].tolist())
    ai_pred = np.array(nlp.AI_EMOTIONS)[emo_probs.argmax(1)]
    classic_pred = np.array([r["label"] for r in nlp.classic_emotion_many(shared["text"].tolist())])
    in_domain = {
        "size": int(len(shared)),
        "transformer": float((ai_pred == truth).mean()),
        "classic": float((classic_pred == truth).mean()),
    }
    print("  spot-check emotion:", {k: round(v, 3) for k, v in spotcheck["emotion_accuracy"].items()})
    print("  spot-check sentiment:", {k: round(v, 3) for k, v in spotcheck["sentiment_accuracy"].items()})
    print("  original test set (5 shared emotions):", {k: round(v, 3) for k, v in in_domain.items() if k != "size"})
    return {"spotcheck": spotcheck, "in_domain": in_domain,
            "transformer_models": {"emotion": nlp.EMOTION_MODEL_ID, "sentiment": nlp.SENTIMENT_MODEL_ID}}


def score_reviews(df):
    """Transformer scores per review, cached and resumable."""
    cols = [f"emo_{e}" for e in nlp.AI_EMOTIONS] + [f"sent_{s}" for s in nlp.SENTIMENTS]
    cache = pd.read_csv(REVIEW_SCORES_CACHE) if os.path.exists(REVIEW_SCORES_CACHE) else pd.DataFrame(columns=["reviewId", *cols])
    todo = df[~df["reviewId"].isin(cache["reviewId"])]
    if len(todo):
        print(f"  transformer scoring {len(todo)} reviews (cached: {len(cache)}) ...")
        todo = todo.iloc[np.argsort(todo["content"].str.len().to_numpy())]
        start = time.time()
        for i in range(0, len(todo), SCORE_CHUNK):
            part = todo.iloc[i:i + SCORE_CHUNK]
            emo, sent = nlp.transformer_scores(part["content"].tolist())
            chunk = pd.DataFrame(np.hstack([emo, sent]).round(5), columns=cols)
            chunk.insert(0, "reviewId", part["reviewId"].to_numpy())
            chunk.to_csv(REVIEW_SCORES_CACHE, mode="a", index=False,
                         header=not os.path.exists(REVIEW_SCORES_CACHE))
            done = i + len(part)
            eta = (time.time() - start) / done * (len(todo) - done) / 60
            print(f"    {done}/{len(todo)}  (~{eta:.0f} min left)", flush=True)
        cache = pd.read_csv(REVIEW_SCORES_CACHE)
    return df.merge(cache.drop_duplicates("reviewId"), on="reviewId", how="left")


def build_review_insights():
    print("Scoring app reviews ...")
    df = pd.read_csv(REVIEWS_CSV)
    df = df.dropna(subset=["content"]).drop_duplicates("reviewId").copy()
    df["content"] = df["content"].astype(str)
    df["app"] = df["appId"].map(APP_NAMES).fillna(df["appId"])
    df["date"] = pd.to_datetime(df["at"], format="%d-%m-%Y %H:%M", errors="coerce")

    lex = df["content"].map(nlp.lexicon_sentiment)
    df["compound"] = lex.map(lambda s: s["vader"]["compound"])
    df["vader_label"] = lex.map(lambda s: s["vader"]["label"])
    df["afinn"] = lex.map(lambda s: s["afinn"]["score"])

    use_ai = nlp.transformers_available()
    if use_ai:
        df = score_reviews(df)
        emo = df[[f"emo_{e}" for e in nlp.AI_EMOTIONS]].to_numpy()
        sent = df[[f"sent_{s}" for s in nlp.SENTIMENTS]].to_numpy()
        df["emotion"] = np.array(nlp.AI_EMOTIONS)[emo.argmax(1)]
        df["sentiment"] = np.array(nlp.SENTIMENTS)[sent.argmax(1)]
        df["score"] = sent[:, 2] - sent[:, 0]
        emotion_labels = nlp.AI_EMOTIONS
    else:
        df["emotion"] = [r["label"] for r in nlp.classic_emotion_many(df["content"].tolist())]
        df["sentiment"] = df["vader_label"]
        df["score"] = df["compound"]
        emotion_labels = nlp.EMOTIONS
    df["replied"] = df["replyContent"].notna()

    clean = df["content"].map(nlp.clean_text)
    stop = set(stopwords.words("english")) | {
        "app", "would", "could", "use", "get", "one", "also", "even", "im", "ive", "dont",
        "cant", "it", "thing", "really", "much", "like", "make", "way", "using", "used",
    }

    def top_words(texts, n=25):
        c = Counter(w for t in texts for w in t.split() if len(w) > 2 and w not in stop)
        return [{"word": w, "count": k} for w, k in c.most_common(n)]

    per_app = []
    for app, g in df.groupby("app"):
        counts = g["sentiment"].value_counts()
        per_app.append({
            "app": app,
            "reviews": int(len(g)),
            "positive": int(counts.get("positive", 0)),
            "neutral": int(counts.get("neutral", 0)),
            "negative": int(counts.get("negative", 0)),
            "avg_score": round(float(g["score"].mean()), 4),
            "avg_compound": round(float(g["compound"].mean()), 4),
            "reply_rate": round(float(g["replied"].mean()), 4),
            "emotions": {e: int((g["emotion"] == e).sum()) for e in emotion_labels},
        })
    per_app.sort(key=lambda r: -r["avg_score"])

    trend = (df.dropna(subset=["date"])
               .assign(month=lambda d: d["date"].dt.to_period("Q").astype(str))
               .groupby("month")
               .agg(avg_score=("score", "mean"), reviews=("score", "size"),
                    negative_share=("sentiment", lambda s: (s == "negative").mean()))
               .reset_index())
    trend = trend[trend["reviews"] >= 30]

    pos_mask, neg_mask = df["sentiment"] == "positive", df["sentiment"] == "negative"
    insights = {
        "models": "transformer" if use_ai else "classic",
        "sentiment_source": "RoBERTa" if use_ai else "VADER",
        "total_reviews": int(len(df)),
        "apps": int(df["app"].nunique()),
        "sentiment_counts": {k: int(v) for k, v in df["sentiment"].value_counts().items()},
        "emotion_labels": emotion_labels,
        "emotion_counts": {e: int((df["emotion"] == e).sum()) for e in emotion_labels},
        "avg_score": round(float(df["score"].mean()), 4),
        "vader_agreement": round(float((df["sentiment"] == df["vader_label"]).mean()), 4),
        "per_app": per_app,
        "trend": trend.round(4).to_dict(orient="records"),
        "top_words": {
            "all": top_words(clean),
            "positive": top_words(clean[pos_mask]),
            "negative": top_words(clean[neg_mask]),
        },
        "score_histogram": np.histogram(df["score"], bins=20, range=(-1, 1))[0].tolist(),
    }
    with open(INSIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)

    cols = ["reviewId", "app", "userName", "content", "thumbsUpCount", "at", "score",
            "compound", "sentiment", "vader_label", "afinn", "emotion", "replyContent"]
    df[cols].round({"score": 4}).to_csv(SCORED_REVIEWS_PATH, index=False)
    print(f"  scored {len(df)} reviews across {insights['apps']} apps")


if __name__ == "__main__":
    test_df, metrics = train_classic_model()
    if nlp.transformers_available():
        metrics["comparison"] = compare_models(test_df)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    build_review_insights()
    print("Artifacts written to", nlp.ARTIFACTS_DIR)
