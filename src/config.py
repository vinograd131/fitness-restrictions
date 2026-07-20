"""Пути и константы проекта."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
OUTPUTS_DIR = ROOT / "outputs"

SEED = 42

TRANSFORMER_ID = "alexyalunin/RuBioRoBERTa"
MAX_LENGTH = 256

# группы, где пропуск дороже ложной тревоги
DANGEROUS = ("ПОЗВОНОЧНИК", "КАРДИО+НЕВРО", "СУСТАВЫ")

# пороги подобраны на dev, см. evaluation/selective.py
CONFIDENCE_THRESHOLD: dict[str, float] = {
    "rubioroberta_ft": 0.80,
    "baseline": 0.75,
}
DEFAULT_THRESHOLD = 0.90
TOP_K = 3

# сколько раз просим уточнить жалобу, прежде чем отдать решение тренеру
MAX_CLARIFY_ATTEMPTS = 3


def threshold_for(model_name: str) -> float:
    return CONFIDENCE_THRESHOLD.get(model_name, DEFAULT_THRESHOLD)


# temperature scaling, подобрана на dev: ECE 0.058 -> 0.026
TEMPERATURE = 1.375
