"""
Train and evaluate. Every run is recorded in `training_runs`.

Usage:
    python train.py                          # default hyperparams
    python train.py --alpha 0.5              # try different smoothing
    python train.py --ngram 1 3              # unigrams..trigrams
    python train.py --min-df 1               # include rare terms
"""
import argparse
import json
import time
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from app.database import sessionLocal, Base, engine
from app.models import Message, TrainingRun, TestPrediction
from app.ml import TextPreprocessor


def build_pipeline(alpha, ngram_min, ngram_max, min_df, max_df, sublinear):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=TextPreprocessor().clean,
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear,
            stop_words="english",
        )),
        ("nb", MultinomialNB(alpha=alpha)),
    ])


def evaluate(pipe, test_rows):
    """Return metrics + per-message predictions."""
    ids     = [r.id      for r in test_rows]
    texts   = [r.content for r in test_rows]
    y_true  = [r.label   for r in test_rows]

    y_pred  = pipe.predict(texts)
    proba   = pipe.predict_proba(texts)
    classes = list(pipe.classes_)
    spam_i  = classes.index("spam")
    scores  = proba[:, spam_i]

    metrics = {
        "accuracy":       float(accuracy_score(y_true, y_pred)),
        "precision_spam": float(precision_score(y_true, y_pred,
                                                pos_label="spam", zero_division=0)),
        "recall_spam":    float(recall_score(y_true, y_pred,
                                             pos_label="spam", zero_division=0)),
        "f1_spam":        float(f1_score(y_true, y_pred,
                                         pos_label="spam", zero_division=0)),
        "f1_weighted":    float(f1_score(y_true, y_pred, average="weighted")),
    }
    per_msg = [
        {
            "message_id": mid,
            "true_label": yt,
            "pred_label": yp,
            "spam_score": float(s),
            "correct":    int(yt == yp),
        }
        for mid, yt, yp, s in zip(ids, y_true, y_pred, scores)
    ]
    cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
    return metrics, per_msg, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha",       type=float, default=0.1)
    ap.add_argument("--ngram",       type=int, nargs=2, default=[1, 2])
    ap.add_argument("--min-df",      type=int, default=2)
    ap.add_argument("--max-df",      type=float, default=0.95)
    ap.add_argument("--sublinear",   action="store_true", default=True)
    ap.add_argument("--output",      default="spam_filter.joblib")
    args = ap.parse_args()

    Base.metadata.create_all(bind=engine)
    db = sessionLocal()

    try:
        # Ensure splits are assigned
        from app.split import assign_splits_once, get_train_rows, get_test_rows
        assign_splits_once(db)

        train = get_train_rows(db)
        test  = get_test_rows(db)

        if len(train) < 20:
            print(f"⚠️  Only {len(train)} training rows. Add more data.")
            return
        if len(test) == 0:
            print("⚠️  No test rows. Run assign_splits_once() first.")
            return

        X_train = [r.content for r in train]
        y_train = [r.label   for r in train]

        print(f"Training on {len(train)} rows "
              f"({y_train.count('spam')} spam / {y_train.count('ham')} ham)")
        print(f"Testing on  {len(test)} rows")

        pipe = build_pipeline(
            alpha=args.alpha,
            ngram_min=args.ngram[0],
            ngram_max=args.ngram[1],
            min_df=args.min_df,
            max_df=args.max_df,
            sublinear=args.sublinear,
        )

        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        metrics, per_msg, cm = evaluate(pipe, test)

        # Save model artifact
        model_version = f"v{int(time.time())}"
        joblib.dump(pipe, args.output)

        # Persist run
        run = TrainingRun(
            model_version=model_version,
            model_path=args.output,
            num_train=len(train),
            num_test=len(test),
            num_train_spam=y_train.count("spam"),
            num_train_ham=y_train.count("ham"),
            num_test_spam=sum(1 for r in test if r.label == "spam"),
            num_test_ham=sum(1 for r in test if r.label == "ham"),
            accuracy=metrics["accuracy"],
            precision_spam=metrics["precision_spam"],
            recall_spam=metrics["recall_spam"],
            f1_spam=metrics["f1_spam"],
            f1_weighted=metrics["f1_weighted"],
            hyperparams=json.dumps({
                "alpha": args.alpha,
                "ngram_range": args.ngram,
                "min_df": args.min_df,
                "max_df": args.max_df,
                "sublinear_tf": args.sublinear,
            }),
        )
        db.add(run)
        db.flush()  # get run.id

        for p in per_msg:
            db.add(TestPrediction(run_id=run.id, **p))
        db.commit()

        # ─── Report ────────────────────────────────────────
        print("=" * 60)
        print(f"Model version : {model_version}")
        print(f"Fit time      : {fit_seconds:.2f}s")
        print("=" * 60)
        for k, v in metrics.items():
            print(f"{k:<16}: {v:.4f}")
        print("\nConfusion Matrix (rows=true, cols=pred, order=ham,spam):")
        print(cm)
        print(f"\n✅ Run #{run.id} saved to training_runs "
              f"({len(per_msg)} per-message predictions).")

        # Show a few misclassifications
        wrong = [p for p in per_msg if p["correct"] == 0]
        if wrong:
            print(f"\n⚠️  {len(wrong)} misclassifications (first 5):")
            for w in wrong[:5]:
                row = db.query(Message).get(w["message_id"])
                preview = row.content[:80] + ("..." if len(row.content) > 80 else "")
                print(f"  true={w['true_label']:<4} pred={w['pred_label']:<4} "
                      f"score={w['spam_score']:.2f}  {preview}")

    finally:
        db.close()


if __name__ == "__main__":
    main()