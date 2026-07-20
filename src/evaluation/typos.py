"""INV-тест на опечатки: человек пишет с телефона и ошибается.

Уровни шума фиксирую заранее и показываю всю кривую, а не одну удобную точку.

  python -m src.evaluation.typos
  python -m src.evaluation.typos --transformer
"""
import argparse
import json
import random

from sklearn.metrics import f1_score

from ..config import REPORTS_DIR as REPORTS, SEED
from ..data import load_xy
from ..models.base import BaseClassifier

RESULT_FILE = REPORTS / "typos.json"
RATES = (0.05, 0.10, 0.20)  # доля слов с опечаткой — зафиксирована ДО прогона
MIN_WORD_LEN = 4  # короткие слова не трогаем: там опечатка убивает слово целиком

# Соседние клавиши ЙЦУКЕН — для реалистичного промаха пальцем.
NEIGHBORS = {
    "й": "цф", "ц": "йуыв", "у": "цкве", "к": "уеап", "е": "кнрп", "н": "егро",
    "г": "ншор", "ш": "гщл", "щ": "шзд", "з": "щхж", "х": "зъэ",
    "ф": "йыя", "ы": "фвяч", "в": "ыаяс", "а": "впсм", "п": "аро",
    "р": "полм", "о": "рлдт", "л": "одж", "д": "лжэ", "ж": "дэ",
    "я": "фчс", "ч": "ясм", "с": "чми", "м": "сит", "и": "мть",
    "т": "иьб", "ь": "тбю", "б": "ью", "ю": "б",
}


def add_typo(word: str, rng: random.Random) -> str:
    """Одна случайная опечатка в слове: перестановка, пропуск, дубль или промах по клавише."""
    i = rng.randrange(len(word))
    op = rng.choice(("swap", "delete", "duplicate", "neighbor"))

    if op == "swap" and i < len(word) - 1:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "duplicate":
        return word[:i] + word[i] + word[i:]
    near = NEIGHBORS.get(word[i].lower())
    if near:
        return word[:i] + rng.choice(near) + word[i + 1:]
    return word


def add_noise(text: str, rate: float, rng: random.Random) -> str:
    """Вносит опечатки в долю `rate` достаточно длинных слов."""
    words = text.split()
    out = []
    for w in words:
        if len(w) >= MIN_WORD_LEN and rng.random() < rate:
            out.append(add_typo(w, rng))
        else:
            out.append(w)
    return " ".join(out)


def main(clf: BaseClassifier, split: str = "test", n_seeds: int = 3) -> dict:
    texts, y_true = load_xy(split)
    clean_f1 = round(float(f1_score(y_true, clf.predict(texts), average="macro")), 4)

    print(f"{clf.name} на {split} (усреднение по {n_seeds} сидам)")
    print(f"{'шум':>6} {'macro-F1':>9} {'±':>7} {'падение':>9}")
    print(f"{'0%':>6} {clean_f1:9.4f} {'—':>7} {'—':>9}")

    rows = []
    for rate in RATES:
        scores = []
        for seed in range(n_seeds):
            rng = random.Random(SEED + seed)
            noisy = [add_noise(t, rate, rng) for t in texts]
            scores.append(float(f1_score(y_true, clf.predict(noisy), average="macro")))
        mean = round(sum(scores) / len(scores), 4)
        std = round((sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5, 4)
        rows.append({"rate": rate, "macro_f1": mean, "std": std,
                     "drop": round(mean - clean_f1, 4)})
        print(f"{rate:6.0%} {mean:9.4f} {std:7.4f} {mean - clean_f1:+9.4f}")

    result = {"model": clf.name, "split": split, "clean_macro_f1": clean_f1,
              "n_seeds": n_seeds, "levels": rows}
    REPORTS.mkdir(exist_ok=True)
    data = json.loads(RESULT_FILE.read_text(encoding="utf-8")) if RESULT_FILE.exists() else {}
    data[clf.name] = result
    RESULT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer", action="store_true", help="считать на прод-модели")
    args = parser.parse_args()

    if args.transformer:
        from ..models.transformer_ft import TransformerClassifier

        model = TransformerClassifier()
    else:
        from .behavioral import fitted_baseline

        model = fitted_baseline()
    main(model)
