"""Диаграмма надёжности: заявленная уверенность против фактической точности.

Точки на диагонали — калибровка идеальная, ниже диагонали — модель самоуверенна.
"""
import json

import matplotlib.pyplot as plt

from ..config import REPORTS_DIR as REPORTS


def main() -> None:
    data = json.loads((REPORTS / "calibration.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="идеальная калибровка")

    for label, color, title in (
        ("before", "#e8590c", f"до (ECE {data['before']['ece']})"),
        ("after", "#3b5bdb", f"после T={data['temperature']} (ECE {data['after']['ece']})"),
    ):
        bins = data[label]["bins"]
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins],
                marker="o", color=color, label=title)

    ax.set_xlabel("заявленная уверенность")
    ax.set_ylabel("фактическая точность")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Калибровка уверенности (temperature scaling)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = REPORTS / "calibration.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"сохранено: {path}")


if __name__ == "__main__":
    main()
