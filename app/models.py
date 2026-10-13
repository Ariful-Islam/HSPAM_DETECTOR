from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, Enum, Index, ForeignKey
)
from sqlalchemy.orm import relationship
from app.database import Base


class Message(Base):
    """Labeled data — training + test, tagged by split."""
    __tablename__ = "messages"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    content    = Column(Text, nullable=False)
    label      = Column(Enum("spam", "ham", name="label_enum"), nullable=False)
    source     = Column(String(32), default="seed")     # seed | user | auto
    split      = Column(
        Enum("train", "test", "unassigned", name="split_enum"),
        default="unassigned", nullable=False,
    )
    user_id    = Column(String(64), nullable=True)      # who submitted feedback
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_messages_label", "label"),
        Index("ix_messages_split", "split"),
        Index("ix_messages_content_label", "content", "label", unique=True),
    )


class TrainingRun(Base):
    """One row per retrain — reproducible experiment log."""
    __tablename__ = "training_runs"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    model_version  = Column(String(64), nullable=False)
    model_path     = Column(String(255), nullable=False)

    # Data counts
    num_train      = Column(Integer, nullable=False)
    num_test       = Column(Integer, nullable=False)
    num_train_spam = Column(Integer, nullable=False)
    num_train_ham  = Column(Integer, nullable=False)
    num_test_spam  = Column(Integer, nullable=False)
    num_test_ham   = Column(Integer, nullable=False)

    # Metrics (only computed on the frozen test split)
    accuracy       = Column(Float, nullable=True)
    precision_spam = Column(Float, nullable=True)
    recall_spam    = Column(Float, nullable=True)
    f1_spam        = Column(Float, nullable=True)
    f1_weighted    = Column(Float, nullable=True)

    # Hyperparameters (JSON string for flexibility)
    hyperparams    = Column(Text, nullable=True)

    created_at     = Column(DateTime, default=datetime.utcnow)

    predictions    = relationship("TestPrediction", back_populates="run",
                                  cascade="all, delete-orphan")


class TestPrediction(Base):
    """Per-message prediction on the test split — the confusion matrix source."""
    __tablename__ = "test_predictions"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    run_id       = Column(Integer, ForeignKey("training_runs.id", ondelete="CASCADE"),
                          nullable=False)
    message_id   = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"),
                          nullable=False)
    true_label   = Column(Enum("spam", "ham", name="true_label_enum"), nullable=False)
    pred_label   = Column(Enum("spam", "ham", name="pred_label_enum"), nullable=False)
    spam_score   = Column(Float, nullable=False)
    correct      = Column(Integer, nullable=False)   # 0/1

    run          = relationship("TrainingRun", back_populates="predictions")

    __table_args__ = (
        Index("ix_testpred_run", "run_id"),
        Index("ix_testpred_correct", "correct"),
    )


class PredictionLog(Base):
    """Every live prediction served (not the test set — production traffic)."""
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    content         = Column(Text, nullable=False)
    predicted_label = Column(Enum("spam", "ham", name="pred_label_enum"), nullable=False)
    spam_score      = Column(Float, nullable=False)
    model_version   = Column(String(64), nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)