"""Проверка утечки через похожие записи.

Точных дублей между сплитами нет, но записи клинические: повторные приёмы одного пациента могут
попасть и в train, и в test — идентификатора пациента в данных нет. Считаю косинусную близость
test к ближайшему train и смотрю, сколько теряет метрика, если выбросить похожие.

  python -m src.evaluation.leakage
"""
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

from ..config import REPORTS_DIR as REPORTS
from ..data import load_xy

RESULT_FILE = REPORTS / "leakage.json"
THRESHOLDS = (0.9, 0.8, 0.7)


def nearest_similarity(x_train: list[str], x_eval: list[str]) -> np.ndarray:
    """Косинусная близость каждого eval-примера к ближайшему train-примеру (tf-idf)."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(x_train + x_eval)
    return cosine_similarity(vec.transform(x_eval), vec.transform(x_train)).max(1)


def main(split: str = "test") -> dict:
    from ..models.baseline import build_model

    x_train, y_train = load_xy("train")
    x_eval, y_eval = load_xy(split)

    best = nearest_similarity(x_train, x_eval)
    pred = build_model().fit(x_train, y_train).predict(x_eval)
    full_f1 = round(float(f1_score(y_eval, pred, average="macro")), 4)

    print(f"близость {split} к ближайшему train: медиана {np.median(best):.3f}, "
          f"95-й перцентиль {np.percentile(best, 95):.3f}")
    print(f"\n{'выборка':28} {'n':>5} {'macro-F1':>9} {'дельта':>8}")
    print(f"{'полный ' + split:28} {len(y_eval):5d} {full_f1:9.4f} {'—':>8}")

    rows = []
    for thr in THRESHOLDS:
        keep = best < thr
        y_keep = [y for y, k in zip(y_eval, keep) if k]
        p_keep = [p for p, k in zip(pred, keep) if k]
        f1 = round(float(f1_score(y_keep, p_keep, average="macro")), 4)
        rows.append({"threshold": thr, "n": len(y_keep), "macro_f1": f1,
                     "delta": round(f1 - full_f1, 4)})
        print(f"{'без похожих >= ' + str(thr):28} {len(y_keep):5d} {f1:9.4f} {f1 - full_f1:+8.4f}")

    print("\nМалая просадка => метрика не раздута похожими записями, сила модели настоящая.")

    result = {"split": split, "full_macro_f1": full_f1,
              "median_similarity": round(float(np.median(best)), 4), "filtered": rows}
    REPORTS.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
