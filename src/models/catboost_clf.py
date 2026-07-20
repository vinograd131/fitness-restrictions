"""CatBoost поверх fastText-эмбеддингов документов."""
import argparse

import numpy as np
from catboost import CatBoostClassifier
from gensim.models import FastText

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
from .base import BaseClassifier
from .fasttext_clf import doc_vectors, tokenize, train_embeddings
from ..mapping import GROUPS

NAME = "catboost"
SEED = 42


def load_or_train_embeddings(train_tokens: list[list[str]]) -> FastText:
    path = MODELS / "fasttext.model"
    if path.exists():
        return FastText.load(str(path))
    return train_embeddings(train_tokens)


class CatBoostGroupClassifier(BaseClassifier):
    """CatBoost поверх fastText-эмбеддингов, модели читаются из models/."""

    name = NAME

    def __init__(self) -> None:
        self._ft = FastText.load(str(MODELS / "fasttext.model"))
        self._clf = CatBoostClassifier()
        self._clf.load_model(str(MODELS / f"{NAME}.cbm"))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        tokens = [tokenize(t) for t in texts]
        return self._clf.predict_proba(doc_vectors(self._ft, tokens))

    @property
    def classes(self) -> list[str]:
        return list(self._clf.classes_)


def main(eval_split: str = "dev") -> None:
    x_train, y_train = load_xy("train")
    x_eval, y_eval = load_xy(eval_split)

    train_tokens = [tokenize(t) for t in x_train]
    eval_tokens = [tokenize(t) for t in x_eval]

    ft = load_or_train_embeddings(train_tokens)
    x_tr = doc_vectors(ft, train_tokens)
    x_ev = doc_vectors(ft, eval_tokens)

    clf = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        random_seed=SEED,
        verbose=False,
    ).fit(x_tr, y_train)
    pred = clf.predict(x_ev).ravel()
    proba = clf.predict_proba(x_ev)

    values = scores(y_eval, pred)
    values["pr_auc"], _ = pr_auc(y_eval, proba, clf.classes_)
    print(f"{NAME} on {eval_split}: {values}")
    print_report(y_eval, pred, labels=list(GROUPS))
    print_dangerous_recall(y_eval, pred)

    save_confusion(y_eval, pred, list(GROUPS), NAME, eval_split)
    save_pr_curves(y_eval, proba, clf.classes_, NAME, eval_split)
    save_report(NAME, eval_split, y_eval, pred, list(GROUPS))
    save_metrics(NAME, eval_split, values)

    MODELS.mkdir(exist_ok=True)
    clf.save_model(str(MODELS / f"{NAME}.cbm"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    main(parser.parse_args().split)
