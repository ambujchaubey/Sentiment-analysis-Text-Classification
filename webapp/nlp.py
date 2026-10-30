"""Shared NLP utilities.

* Lexicon sentiment: VADER + AFINN (from PreProc_lexicon_based.ipynb)
* Classic emotion model: TF-IDF + Logistic Regression (from text_classification.ipynb)
* Transformer models (primary): RoBERTa sentiment and DistilRoBERTa emotion, pretrained on
  tweets / reviews / dialogue, so they handle real app-review language much better.

Set SENTISCOPE_FAST=1 to skip the transformer models (lexicon + classic only).
"""
import os
import re
from functools import lru_cache

import joblib
import nltk
import numpy as np
from afinn import Afinn
from nltk.sentiment.vader import SentimentIntensityAnalyzer

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "emotion_model.joblib")

EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
AI_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENTS = ["negative", "neutral", "positive"]
EMOTION_EMOJI = {
    "sadness": "😢", "joy": "😄", "love": "❤️", "anger": "😠",
    "fear": "😨", "surprise": "😲", "disgust": "🤢", "neutral": "😐",
}

EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_TOKENS = 128

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

_vader = SentimentIntensityAnalyzer()
_afinn = Afinn()

_URL_RE = re.compile(r"http[s]?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NUMWORD_RE = re.compile(r"\b\w*\d\w*\b")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Same cleaning steps as the notebooks: lowercase, strip links/HTML/numbers/punctuation."""
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NUMWORD_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return " ".join(text.split())


def label_from_compound(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def label_from_afinn(score: float) -> str:
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def lexicon_sentiment(text: str) -> dict:
    vader = _vader.polarity_scores(str(text))
    afinn_score = _afinn.score(str(text))
    return {
        "vader": {**vader, "label": label_from_compound(vader["compound"])},
        "afinn": {"score": afinn_score, "label": label_from_afinn(afinn_score)},
    }


# ---------- classic TF-IDF model ----------

@lru_cache(maxsize=1)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Emotion model not found. Run `python webapp/build_artifacts.py` first."
        )
    return joblib.load(MODEL_PATH)


def classic_emotion_many(texts: list[str]) -> list[dict]:
    model = load_model()
    probs = model.predict_proba([clean_text(t) for t in texts])
    labels = [EMOTIONS[int(c)] for c in model.classes_]
    return [_scores_to_result(dict(zip(labels, map(float, row)))) for row in probs]


def _scores_to_result(scores: dict) -> dict:
    top = max(scores, key=scores.get)
    return {
        "label": top,
        "emoji": EMOTION_EMOJI[top],
        "confidence": scores[top],
        "scores": dict(sorted(scores.items(), key=lambda kv: -kv[1])),
    }


# ---------- transformer models ----------

class _Classifier:
    def __init__(self, model_id: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        self.labels = [self.model.config.id2label[i].lower()
                       for i in range(self.model.config.num_labels)]

    def predict(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return class probabilities; batches texts of similar length to minimise padding."""
        import torch

        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                idx = order[i:i + batch_size]
                enc = self.tokenizer([texts[j] for j in idx], truncation=True,
                                     max_length=MAX_TOKENS, padding=True, return_tensors="pt")
                out[idx] = torch.softmax(self.model(**enc).logits, dim=-1).numpy()
        return out


@lru_cache(maxsize=1)
def _transformers():
    if os.environ.get("SENTISCOPE_FAST") == "1":
        return None
    try:
        import torch

        torch.set_num_threads(os.cpu_count() or 4)
        return {"emotion": _Classifier(EMOTION_MODEL_ID), "sentiment": _Classifier(SENTIMENT_MODEL_ID)}
    except Exception as exc:  # transformers/torch missing or model download failed
        print(f"[nlp] Transformer models unavailable, using lexicon + classic models: {exc}")
        return None


def transformers_available() -> bool:
    return _transformers() is not None


def transformer_scores(texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """(emotion_probs[n, 7], sentiment_probs[n, 3]) in AI_EMOTIONS / SENTIMENTS order."""
    models = _transformers()
    texts = [str(t) for t in texts]
    emo = models["emotion"].predict(texts, batch_size)
    sent = models["sentiment"].predict(texts, batch_size)
    emo = emo[:, [models["emotion"].labels.index(l) for l in AI_EMOTIONS]]
    sent = sent[:, [models["sentiment"].labels.index(l) for l in SENTIMENTS]]
    return emo, sent


# ---------- combined analysis ----------

def analyze_many(texts: list[str], fast: bool = False) -> list[dict]:
    texts = [str(t) for t in texts]
    classic = classic_emotion_many(texts)
    use_ai = not fast and transformers_available()
    if use_ai:
        emo_probs, sent_probs = transformer_scores(texts)

    results = []
    for i, text in enumerate(texts):
        lex = lexicon_sentiment(text)
        sentiment = {**lex, "label": lex["vader"]["label"], "source": "vader"}
        emotion = classic[i]
        if use_ai:
            roberta = dict(zip(SENTIMENTS, map(float, sent_probs[i])))
            label = max(roberta, key=roberta.get)
            sentiment.update(
                roberta={"scores": roberta, "label": label, "confidence": roberta[label],
                         "polarity": roberta["positive"] - roberta["negative"]},
                label=label, source="roberta",
            )
            emotion = _scores_to_result(dict(zip(AI_EMOTIONS, map(float, emo_probs[i]))))
        results.append({
            "text": text,
            "sentiment": sentiment,
            "emotion": emotion,
            "emotion_classic": classic[i],
            "models": "transformer" if use_ai else "classic",
        })
    return results


def analyze(text: str) -> dict:
    return analyze_many([text])[0]
