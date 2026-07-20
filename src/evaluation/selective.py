"""Порог уверенности: лучше переспросить, чем дать неверное ограничение.

Порог подбираю на dev, а не на глаз: строю точность и покрытие от порога и беру наименьший
порог, где точность >= TARGET_ACC.

  python -m src.evaluation.selective
  python -m src.evaluation.selective --transformer
"""
import argparse
import json

import numpy as np

from ..config import REPORTS_DIR as REPORTS
from ..data import load_xy
from ..models.base import BaseClassifier

RESULT_FILE = REPORTS / "selective.json"
TARGET_ACC = 0.95  # какую точность хотим на принятых предсказаниях
GRID = np.round(np.arange(0.30, 0.96, 0.05), 2)


def analyze(clf: BaseClassifier, split: str = "dev") -> dict:
    texts, y_true = load_xy(split)
    proba = np.asarray(clf.predict_proba(texts))
    conf = proba.max(1)
    pred = [clf.classes[i] for i in proba.argmax(1)]
    correct = np.array([p == t for p, t in zip(pred, y_true)])

    rows = []
    for thr in GRID:
        taken = conf >= thr
        coverage = float(taken.mean())
        acc = float(correct[taken].mean()) if taken.any() else 1.0
        rows.append({"threshold": float(thr), "coverage": round(coverage, 4),
                     "accuracy": round(acc, 4)})

    # наименьший порог, дающий нужную точность (максимальное покрытие при этой точности)
    ok = [r for r in rows if r["accuracy"] >= TARGET_ACC]
    chosen = min(ok, key=lambda r: r["threshold"]) if ok else max(rows, key=lambda r: r["accuracy"])

    print(f"{clf.name} на {split}: точность без порога = {correct.mean():.4f}")
    print(f"{'порог':>6} {'покрытие':>9} {'точность':>9}")
    for r in rows:
        mark = "  <- выбран" if r["threshold"] == chosen["threshold"] else ""
        print(f"{r['threshold']:6.2f} {r['coverage']:9.3f} {r['accuracy']:9.3f}{mark}")
    print(
        f"\nПорог {chosen['threshold']}: точность {chosen['accuracy']:.3f} "
        f"на {chosen['coverage']:.1%} жалоб, остальные {1 - chosen['coverage']:.1%} — на уточнение."
    )

    result = {"model": clf.name, "split": split, "target_accuracy": TARGET_ACC,
              "chosen": chosen, "grid": rows}
    REPORTS.mkdir(exist_ok=True)
    data = json.loads(RESULT_FILE.read_text(encoding="utf-8")) if RESULT_FILE.exists() else {}
    data[clf.name] = result
    RESULT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(use_transformer: bool = False) -> None:
    if use_transformer:
        from ..models.transformer_ft import TransformerClassifier

        clf = TransformerClassifier()
    else:
        from .behavioral import fitted_baseline

        clf = fitted_baseline()
    analyze(clf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer", action="store_true", help="считать на прод-модели")
    main(parser.parse_args().transformer)
