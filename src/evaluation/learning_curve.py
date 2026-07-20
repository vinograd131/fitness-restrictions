"""Кривая обучения: dev macro-F1 от размера train.

Плато означает, что упираюсь не в объём данных. Считаю на baseline, усредняю по сидам.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from ..models.baseline import build_model
from ..data import load_xy

from ..config import REPORTS_DIR as REPORTS
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]


def main() -> None:
    x_train, y_train = load_xy("train")
    x_dev, y_dev = load_xy("dev")
    x_train = np.array(x_train, dtype=object)
    y_train = np.array(y_train)

    sizes, means, stds = [], [], []
    for frac in FRACTIONS:
        scores, n = [], 0
        for seed in SEEDS:
            if frac < 1.0:
                splitter = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
                idx, _ = next(splitter.split(x_train, y_train))
                xs, ys = list(x_train[idx]), list(y_train[idx])
            else:
                xs, ys = list(x_train), list(y_train)
            model = build_model().fit(xs, ys)
            scores.append(f1_score(y_dev, model.predict(x_dev), average="macro"))
            n = len(ys)
        sizes.append(n)
        means.append(float(np.mean(scores)))
        stds.append(float(np.std(scores)))
        print(f"train={n:4d}  dev macro-F1={means[-1]:.4f} ± {stds[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, color="#3b5bdb")
    ax.set_xlabel("размер train (примеров)")
    ax.set_ylabel("dev macro-F1")
    ax.set_title("Кривая обучения (tf-idf + LogReg)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = REPORTS / "learning_curve.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"сохранено: {path}")


if __name__ == "__main__":
    main()
