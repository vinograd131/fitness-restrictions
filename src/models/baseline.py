"""Baseline: tf-idf + логистическая регрессия.

Лемматизация pymorphy3, чтобы словоформы (болит/болела/боли) не расползались по разным
признакам.
"""
import argparse
import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..config import MODELS_DIR as MODELS
from ..data import load_xy
from ..evaluation.metrics import (
    pr_auc,
    print_dangerous_recall,
    print_report,
    save_confusion,
    save_metrics,
    save_pr_curves,
    save_report,
    scores,
)
from ..mapping import GROUPS
from .base import BaseClassifier

NAME = "baseline"

_TOKEN = re.compile(r"[а-яёa-z]+")
_morph = None
_lemma_cache: dict[str, str] = {}


def lemmatize(text: str) -> str:
    """Лемматизация токенов с кэшем — идёт в TfidfVectorizer как preprocessor."""
    global _morph
    if _morph is None:
        import pymorphy3

        _morph = pymorphy3.MorphAnalyzer()
    out = []
    for tok in _TOKEN.findall(text.lower()):
        lem = _lemma_cache.get(tok)
        if lem is None:
            lem = _morph.parse(tok)[0].normal_form
            _lemma_cache[tok] = lem
        out.append(lem)
    return " ".join(out)


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=lemmatize, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, C=10, class_weight="balanced")),
    ])


class TfidfClassifier(BaseClassifier):
    """tf-idf + логрег: быстрый и интерпретируемый."""

    name = NAME

    def __init__(self) -> None:
        self._pipe = build_model()

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfClassifier":
        self._pipe.fit(texts, labels)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self._pipe.predict_proba(texts)

    @property
    def classes(self) -> list[str]:
        return list(self._pipe.classes_)


def main(eval_split: str = "dev") -> None:
    x_train, y_train = load_xy("train")
    x_eval, y_eval = load_xy(eval_split)

    model = build_model().fit(x_train, y_train)
    pred = model.predict(x_eval)
    proba = model.predict_proba(x_eval)

    values = scores(y_eval, pred)
    values["pr_auc"], _ = pr_auc(y_eval, proba, model.classes_)
    print(f"{NAME} on {eval_split}: {values}")
    print_report(y_eval, pred, labels=list(GROUPS))
    print_dangerous_recall(y_eval, pred)

    save_confusion(y_eval, pred, list(GROUPS), NAME, eval_split)
    save_pr_curves(y_eval, proba, model.classes_, NAME, eval_split)
    save_report(NAME, eval_split, y_eval, pred, list(GROUPS))
    save_metrics(NAME, eval_split, values)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / f"{NAME}.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    main(parser.parse_args().split)
