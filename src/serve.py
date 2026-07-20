"""REST-сервис: жалоба -> группа ограничений и рекомендации.

  GET  /ping /health /groups
  POST /predict  {"text": "..."}

MODEL=transformer (по умолчанию) или MODEL=baseline для запуска без torch.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import TOP_K, threshold_for
from .mapping import CLARIFY_MESSAGE, DISCLAIMER, GROUPS, RESTRICTIONS
from .models.base import BaseClassifier

MODEL_KIND = os.environ.get("MODEL", "transformer")

_state: dict = {}


def build_classifier(kind: str = MODEL_KIND) -> BaseClassifier:
    if kind == "baseline":
        from .evaluation.behavioral import fitted_baseline

        return fitted_baseline()

    from .models.transformer_ft import TransformerClassifier

    return TransformerClassifier()


def _get_classifier() -> BaseClassifier:
    if "clf" not in _state:
        _state["clf"] = build_classifier()
    return _state["clf"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # холостой проход на старте: без него первый запрос ждёт ленивую инициализацию torch (~20 с)
    _get_classifier().predict_proba(["прогрев"])
    yield


app = FastAPI(title="anamnes — классификатор фитнес-ограничений", lifespan=lifespan)


class Complaint(BaseModel):
    text: str = Field(..., min_length=1, examples=["болит поясница, отдаёт в ногу"])


class Candidate(BaseModel):
    group: str
    confidence: float


class Prediction(BaseModel):
    group: str
    summary: str
    forbidden: list[str]
    caution: list[str]
    equipment: list[str]
    confidence: float
    needs_clarification: bool
    message: str | None = None
    alternatives: list[Candidate] = []
    disclaimer: str = DISCLAIMER


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_KIND, "loaded": "clf" in _state}


@app.get("/groups")
def groups():
    return [{"group": g, **RESTRICTIONS[g]} for g in GROUPS]


@app.post("/predict", response_model=Prediction)
def predict(item: Complaint):
    clf = _get_classifier()
    proba = clf.predict_proba([item.text])[0]
    idx = int(proba.argmax())
    group = clf.classes[idx]
    confidence = round(float(proba[idx]), 4)
    rec = RESTRICTIONS[group]

    # ниже порога не настаиваем на ответе, а просим уточнить жалобу
    uncertain = confidence < threshold_for(clf.name)
    alternatives = []
    if uncertain:
        top = sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)[:TOP_K]
        alternatives = [
            Candidate(group=clf.classes[i], confidence=round(float(proba[i]), 4)) for i in top
        ]

    return Prediction(
        group=group,
        summary=rec["summary"],
        forbidden=rec["forbidden"],
        caution=rec["caution"],
        equipment=rec["equipment"],
        confidence=confidence,
        needs_clarification=uncertain,
        message=CLARIFY_MESSAGE if uncertain else None,
        alternatives=alternatives,
    )
