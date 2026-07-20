"""Поведенческие тесты по CheckList (Ribeiro et al., 2020).

MFT — короткие однозначные жалобы, собираю шаблонами из лексиконов симптомов.
INV (перефраз) лежит в robustness.py.
"""
import json


from ..config import REPORTS_DIR as REPORTS
from ..data import load_xy
from ..models.base import BaseClassifier

RESULTS_FILE = REPORTS / "behavioral.json"

LEXICON: dict[str, list[str]] = {
    "ПОЗВОНОЧНИК": [
        "боль в пояснице", "боль в спине", "боль в шее", "прострел в пояснице",
        "скованность в спине", "боль в поясничном отделе",
    ],
    "СУСТАВЫ": [
        "боль в колене", "боль в коленном суставе", "боль в плечевом суставе",
        "отёк сустава", "хруст в суставах", "боль в суставах пальцев",
    ],
    "КАРДИО+НЕВРО": [
        "повышение давления", "учащённое сердцебиение", "перебои в работе сердца",
        "боль в области сердца", "онемение пальцев рук", "дрожь в руках",
    ],
    "КОЖА": [
        "сыпь на коже", "зуд кожи", "высыпания на коже", "шелушение кожи",
        "покраснение кожи", "угревая сыпь",
    ],
    "ДЫХАТЕЛЬНЫЕ": [
        "насморк", "заложенность носа", "боль в горле", "кашель",
        "чихание", "першение в горле",
    ],
    "ЖКТ/БРЮШНЫЕ": [
        "изжога", "тошнота", "боль в животе", "вздутие живота", "отрыжка",
        "тяжесть в желудке",
    ],
    "МОЧЕПОЛОВЫЕ/ТАЗ": [
        "рези при мочеиспускании", "частое мочеиспускание", "боль при мочеиспускании",
        "мутная моча", "недержание мочи", "болезненные менструации",
    ],
    "ЭНДОКРИН/МЕТАБОЛ": [
        "постоянная жажда", "сухость во рту", "набор веса", "избыточный вес",
        "повышенный сахар в крови", "увеличение щитовидной железы",
    ],
}

TEMPLATES: list[str] = [
    "{s}", "периодически {s}", "последнее время {s}", "постоянно {s}",
    "часто {s}", "уже неделю {s}",
]


def build_mft_cases() -> list[tuple[str, str]]:
    cases, seen = [], set()
    for group, symptoms in LEXICON.items():
        for s in symptoms:
            for t in TEMPLATES:
                text = t.format(s=s)
                if text not in seen:
                    seen.add(text)
                    cases.append((text, group))
    return cases


MFT_CASES: list[tuple[str, str]] = build_mft_cases()


def _save(test: str, model_name: str, values: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    data = {}
    if RESULTS_FILE.exists():
        data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    data.setdefault(test, {})[model_name] = values
    RESULTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_mft(clf: BaseClassifier) -> dict:
    """MFT: точность на однозначных жалобах, общая и по группам. Работает с любой моделью."""
    texts = [t for t, _ in MFT_CASES]
    gold = [g for _, g in MFT_CASES]
    pred = clf.predict(texts)

    correct = sum(p == g for p, g in zip(pred, gold))
    per_group = {}
    for grp in sorted(set(gold)):
        idx = [i for i, g in enumerate(gold) if g == grp]
        per_group[grp] = round(sum(pred[i] == grp for i in idx) / len(idx), 3)

    values = {"accuracy": round(correct / len(gold), 4), "n": len(gold), "per_group": per_group}
    print(f"[MFT] {clf.name}: accuracy {values['accuracy']} ({correct}/{len(gold)})")
    for grp, acc in per_group.items():
        flag = "" if acc == 1.0 else "  <-"
        print(f"    {grp:18s} {acc:.3f}{flag}")
    _save("mft", clf.name, values)
    return values


def fitted_baseline() -> BaseClassifier:
    """tf-idf + LogReg, обученный на train."""
    from ..models.baseline import TfidfClassifier

    x_train, y_train = load_xy("train")
    return TfidfClassifier().fit(x_train, y_train)


def main() -> None:
    run_mft(fitted_baseline())


if __name__ == "__main__":
    main()
