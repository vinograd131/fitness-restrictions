"""График MFT: точность на однозначных жалобах, baseline vs трансформер."""
import json

import matplotlib.pyplot as plt
import numpy as np

from ..config import REPORTS_DIR as REPORTS
MODELS = {
    "baseline": "tf-idf + LogReg",
    "fasttext": "fastText",
    "catboost": "CatBoost",
    "rubioroberta_ft": "RuBioRoBERTa файнтюн",
}
COLORS = {"baseline": "#e8590c", "fasttext": "#2f9e44", "catboost": "#f08c00", "rubioroberta_ft": "#0891b2"}
BLUE, ORANGE = "#3b5bdb", "#e8590c"


def plot_by_group() -> None:
    data = json.loads((REPORTS / "behavioral.json").read_text(encoding="utf-8"))["mft"]
    order = sorted(data["baseline"]["per_group"], key=lambda g: data["rubioroberta_ft"]["per_group"][g])
    y = np.arange(len(order))
    n = len(MODELS)
    height = 0.8 / n

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, name in enumerate(MODELS):
        vals = [data[name]["per_group"][g] for g in order]
        offset = (i - (n - 1) / 2) * height
        ax.barh(y + offset, vals, height, color=COLORS[name], label=MODELS[name])
    ax.axvline(0.70, color="#94a3b8", ls="--", lw=1)
    ax.text(0.71, y[-1] + 0.5, "порог 0.70", color="#64748b", fontsize=9, va="center")

    ax.set_yticks(y, order)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("accuracy (по 36 кейсов на группу)")
    ax.set_title("MFT — точность по группам: где проседает каждая модель")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path = REPORTS / "mft_by_group.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"сохранено: {path}")


def main() -> None:
    plot_by_group()
    data = json.loads((REPORTS / "behavioral.json").read_text(encoding="utf-8"))["mft"]
    names = list(MODELS)
    acc = [data[m]["accuracy"] for m in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, acc, 0.55, color=[COLORS[m] for m in names])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=11)
    ax.set_xticks(x, [MODELS[m] for m in names], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("точность")
    ax.set_title("MFT — короткие жалобы (CheckList)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = REPORTS / "behavioral.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"сохранено: {path}")


if __name__ == "__main__":
    main()
