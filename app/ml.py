import re
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from app.config import settings


class TextPreprocessor:
    def __init__(self):
        self.url_pattern    = re.compile(r"http\S+|www\.\S+")
        self.email_pattern  = re.compile(r"\S+@\S+")
        self.number_pattern = re.compile(r"\b\d+\b")
        self.non_alpha      = re.compile(r"[^a-z\s]")
        self.whitespace     = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        text = text.lower()
        text = self.url_pattern.sub(" url ", text)
        text = self.email_pattern.sub(" email ", text)
        text = self.number_pattern.sub(" num ", text)
        text = self.non_alpha.sub(" ", text)
        text = self.whitespace.sub(" ", text).strip()
        return text


class ModelStore:
    """Thread-safe in-memory cache of the loaded pipeline."""
    _lock = __import__("threading").Lock()

    def __init__(self, path: str):
        self.path = Path(path)
        self.pipeline = None
        self.version = "unloaded"
        self.load()

    def load(self):
        with self._lock:
            if self.path.exists():
                self.pipeline = joblib.load(self.path)
                self.version = f"v{int(self.path.stat().st_mtime)}"
            else:
                self.pipeline = None
                self.version = "no-model"

    def is_ready(self):
        return self.pipeline is not None


model_store = ModelStore(settings.MODEL_PATH)


def predict_with_explanation(message: str, top_k: int = 8) -> dict:
    if not model_store.is_ready():
        raise RuntimeError("Model not trained. Run `python train.py` first.")

    pipe    = model_store.pipeline
    proba   = pipe.predict_proba([message])[0]
    classes = list(pipe.classes_)
    spam_i  = classes.index("spam")
    spam_score = float(proba[spam_i])

    vec = pipe.named_steps["tfidf"]
    nb  = pipe.named_steps["nb"]
    X   = vec.transform([message])
    log_diff = nb.feature_log_prob_[spam_i] - nb.feature_log_prob_[1 - spam_i]
    names = vec.get_feature_names_out()

    present = X.nonzero()[1]
    contribs = [(names[i], float(X[0, i] * log_diff[i])) for i in present]
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "label": "spam" if spam_score >= 0.5 else "ham",
        "spam_score": round(spam_score, 4),
        "ham_score": round(1 - spam_score, 4),
        "top_tokens": [{"token": t, "weight": round(w, 4)}
                       for t, w in contribs[:top_k]],
    }