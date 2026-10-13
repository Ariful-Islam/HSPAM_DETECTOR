from typing import List
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10_000)


class TokenContribution(BaseModel):
    token: str
    weight: float


class PredictResponse(BaseModel):
    label: str
    spam_score: float
    ham_score: float
    top_tokens: List[TokenContribution]


class FeedbackRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10_000)
    label: str = Field(..., pattern="^(spam|ham)$")
    user_id: str | None = None


class FeedbackResponse(BaseModel):
    id: int
    label: str
    status: str
    split: str