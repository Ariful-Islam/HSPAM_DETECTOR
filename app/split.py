from collections import defaultdict
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from app.models import Message


def assign_splits_once(
    db: Session,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Assign train/test to every row where split='unassigned'.
    STRATIFIED by label. Never touches rows already assigned.
    Returns counts.
    """
    unassigned = (
        db.query(Message)
        .filter(Message.split == "unassigned")
        .all()
    )
    if not unassigned:
        return _counts(db)

    texts  = [m.content for m in unassigned]
    labels = [m.label   for m in unassigned]

    # Stratify only if both classes present in sufficient quantity
    strat = labels if len(set(labels)) == 2 and len(labels) >= 20 else None

    train_idx, test_idx = train_test_split(
        range(len(unassigned)),
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    for i in train_idx:
        unassigned[i].split = "train"
    for i in test_idx:
        unassigned[i].split = "test"

    db.commit()
    return _counts(db)


def _counts(db: Session) -> dict:
    rows = db.query(Message.split, Message.label).all()
    out = defaultdict(int)
    for split, label in rows:
        out[f"{split}_{label}"] += 1
        out[split] += 1
    return dict(out)


def get_train_rows(db: Session):
    return db.query(Message.content, Message.label).filter(
        Message.split == "train"
    ).all()


def get_test_rows(db: Session):
    return db.query(Message.id, Message.content, Message.label).filter(
        Message.split == "test"
    ).all()