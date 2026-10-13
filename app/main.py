from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import Base, engine, get_db
from app.models import Message, PredictionLog, TrainingRun, TestPrediction
from app.ml import predict_with_explanation, model_store
from app.schemas import PredictRequest, PredictResponse, FeedbackRequest, FeedbackResponse


app = FastAPI(title="Spam Detector", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": model_store.is_ready(),
        "model_version": model_store.version,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    try:
        result = predict_with_explanation(req.message)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    db.add(PredictionLog(
        content=req.message,
        predicted_label=result["label"],
        spam_score=result["spam_score"],
        model_version=model_store.version,
    ))
    db.commit()
    return result


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """New feedback → `unassigned`. Never enters test split."""
    existing = (
        db.query(Message)
        .filter(Message.content == req.message, Message.label == req.label)
        .first()
    )
    if existing:
        return {"id": existing.id, "label": existing.label,
                "status": "already_exists", "split": existing.split}

    msg = Message(content=req.message, label=req.label,
                  source="user", split="unassigned")
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return {"id": msg.id, "label": msg.label,
            "status": "stored", "split": msg.split}


@app.get("/runs")
def list_runs(db: Session = Depends(get_db), limit: int = 20):
    runs = (db.query(TrainingRun)
              .order_by(TrainingRun.id.desc())
              .limit(limit).all())
    return [{
        "id": r.id,
        "model_version": r.model_version,
        "accuracy": r.accuracy,
        "f1_spam": r.f1_spam,
        "f1_weighted": r.f1_weighted,
        "num_train": r.num_train,
        "num_test": r.num_test,
        "created_at": r.created_at.isoformat(),
        "hyperparams": r.hyperparams,
    } for r in runs]


@app.get("/runs/{run_id}")
def run_detail(run_id: int, db: Session = Depends(get_db)):
    run = db.query(TrainingRun).get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    preds = (db.query(TestPrediction)
               .filter(TestPrediction.run_id == run_id)
               .all())

    # Join message content for display
    msgs = {m.id: m.content
            for m in db.query(Message)
                       .filter(Message.id.in_([p.message_id for p in preds]))
                       .all()}

    return {
        "run": {
            "id": run.id,
            "model_version": run.model_version,
            "accuracy": run.accuracy,
            "precision_spam": run.precision_spam,
            "recall_spam": run.recall_spam,
            "f1_spam": run.f1_spam,
            "f1_weighted": run.f1_weighted,
            "num_train": run.num_train,
            "num_test": run.num_test,
            "hyperparams": run.hyperparams,
        },
        "predictions": [{
            "message_id": p.message_id,
            "content": msgs.get(p.message_id, "")[:120],
            "true_label": p.true_label,
            "pred_label": p.pred_label,
            "spam_score": p.spam_score,
            "correct": bool(p.correct),
        } for p in preds],
    }


@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    row = dict(
        total     = db.query(Message).count(),
        spam      = db.query(Message).filter(Message.label == "spam").count(),
        ham       = db.query(Message).filter(Message.label == "ham").count(),
        train     = db.query(Message).filter(Message.split == "train").count(),
        test      = db.query(Message).filter(Message.split == "test").count(),
        unassigned= db.query(Message).filter(Message.split == "unassigned").count(),
        runs      = db.query(TrainingRun).count(),
        preds     = db.query(PredictionLog).count(),
    )
    return row