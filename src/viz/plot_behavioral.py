"""График MFT: точность на однозначных жалобах, baseline vs трансформер."""
import json

import matplotlib.pyplot as plt
import numpy as np

from ..config import REPORTS_DIR as REPORTS
MODELS = {"baseline": "tf-idf + LogReg", "rubioroberta_ft": "RuBioRoBERTa файнтюн"}
BLUE, ORANGE = "#3b5bdb", "#e8590c"
C_BASE, C_FT = "#e8590c", "#0891b2"


def plot_by_group() -> None:
    data = json.loads((REPORTS / "behavioral.json").read_text(encoding="utf-8"))["mft"]
    base, ft = data["baseline"]["per_group"], data["rubioroberta_ft"]["per_group"]
    order = sorted(base, key=lambda g: ft[g])
    y = np.arange(len(order))
    b = np.array([base[g] for g in order])
    f = np.array([ft[g] for g in order])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hlines(y, b, f, color="#cbd5e1", lw=2.5, zorder=1)
    ax.scatter(b, y, s=90, color=C_BASE, zorder=2, label="baseline")
    ax.scatter(f, y, s=90, color=C_FT, zorder=2, label="трансформер")
    ax.axvline(0.70, color="#94a3b8", ls="--", lw=1)
    ax.text(0.71, 0.15, "порог 0.70", color="#64748b", fontsize=9, va="center")
    for yi, g in zip(y, order):
        lo, hi = sorted([base[g], ft[g]])
        ax.text(lo - 0.012, yi, f"{lo:.2f}", ha="right", va="center", fontsize=8.5, color="#334155")
        if hi > lo:
            ax.text(hi + 0.012, yi, f"{hi:.2f}", ha="left", va="center", fontsize=8.5, color="#334155")

    ax.set_yticks(y, order)
    ax.set_xlim(0.30, 1.06)
    ax.set_xlabel("accuracy (по 36 кейсов на группу)")
    ax.set_title("MFT — точность по группам: где проседает каждая модель")
    ax.legend(loc="upper left", framealpha=0.9)
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

    fig, ax = plt.subplots(figsize=(6.5, 5))
    bars = ax.bar(x, acc, 0.5, color=[BLUE, ORANGE])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=11)
    ax.set_xticks(x, [MODELS[m] for m in names])
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
