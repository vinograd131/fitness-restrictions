"""Temperature scaling (Guo et al., 2017): калибрую уверенность трансформера.

Логиты делю на T перед softmax, T подбираю на dev по NLL. argmax не меняется — точность та же,
честнее становится только уверенность. Качество меряю через ECE.

  python -m src.evaluation.calibration    # нужен torch
"""
import json

import numpy as np
from scipy.optimize import minimize_scalar

from ..config import REPORTS_DIR as REPORTS
from ..data import load_xy

RESULT_FILE = REPORTS / "calibration.json"
N_BINS = 10


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(temperature: float, logits: np.ndarray, y_idx: np.ndarray) -> float:
    proba = _softmax(logits / temperature)
    return float(-np.mean(np.log(proba[np.arange(len(y_idx)), y_idx] + 1e-12)))


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS):
    """ECE + разбивка по корзинам (для диаграммы надёжности)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        avg_conf = float(conf[mask].mean())
        acc = float(correct[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(acc - avg_conf)
        bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "confidence": round(avg_conf, 4),
                     "accuracy": round(acc, 4), "share": round(weight, 4)})
    return round(ece, 4), bins


def main(split: str = "dev") -> dict:
    from ..models.transformer_ft import LABEL2ID, TransformerClassifier

    clf = TransformerClassifier(temperature=1.0)  # калибруем сырую модель
    texts, y_true = load_xy(split)
    logits = clf.predict_logits(texts)
    y_idx = np.array([LABEL2ID[y] for y in y_true])

    res = minimize_scalar(_nll, bounds=(0.5, 5.0), method="bounded", args=(logits, y_idx))
    temperature = round(float(res.x), 3)

    out = {"model": clf.name, "split": split, "temperature": temperature}
    for label, t in (("before", 1.0), ("after", temperature)):
        proba = _softmax(logits / t)
        conf = proba.max(1)
        correct = proba.argmax(1) == y_idx
        ece, bins = expected_calibration_error(conf, correct)
        out[label] = {"ece": ece, "mean_confidence": round(float(conf.mean()), 4),
                      "accuracy": round(float(correct.mean()), 4), "bins": bins}

    print(f"подобранная температура T = {temperature}")
    print(f"{'':8} {'ECE':>8} {'ср. уверенность':>16} {'точность':>10}")
    for label in ("before", "after"):
        d = out[label]
        print(f"{label:8} {d['ece']:8.4f} {d['mean_confidence']:16.4f} {d['accuracy']:10.4f}")
    print("\nargmax не меняется -> точность та же, честнее стала только уверенность.")

    REPORTS.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"сохранено: {RESULT_FILE}")
    return out


if __name__ == "__main__":
    main()
