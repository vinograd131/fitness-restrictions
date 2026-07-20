"""Загрузка сплитов и подготовка пар (текст, группа)."""
import json
import re

from .config import DATA_DIR
from .mapping import group_of, is_dropped

_WS = re.compile(r"\s+")

# обороты приёма без симптома: «жалоб нет», «повторный осмотр»
_ADMIN = re.compile(
    r"жалоб\w*\s*(нет|прежни\w*|не\s*предъявл\w*|не\s*предьявл\w*)"
    r"|активно\s*не\s*предъявляет|не\s*предъявляет"
    r"|повторн\w*\s*(прием|приём|осмотр|консультац\w*)|визит\s*повторн\w*"
    r"|плановая\s*консультац\w*|контрольн\w*\s*осмотр\w*"
    r"|состояние\s*(без\s*динамики|улучшилось|стабильное)"
    r"|(общее\s*)?самочувствие\s*(нормальн\w*|улучшилось)"
    r"|на\s*момент\s*осмотра( нет)?|с\s*жалобами\s*ознакомлена"
    r"|прежние|прием|\*дата\*|нестабильный$"
)
_FILLER = {"в", "на", "с", "и", "не", "нет", "по", "после", "о", "у", "-"}
_MAX_ADMIN_WORDS = 6
_NON_WORD = re.compile(r"[^а-яёa-z0-9 ]")


def load_split(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def normalize(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def is_uninformative(text: str) -> bool:
    """Короткая запись, где после снятия оборотов симптома не осталось."""
    norm = normalize(text)
    if len(norm.split()) > _MAX_ADMIN_WORDS or not _ADMIN.search(norm):
        return False
    rest = _NON_WORD.sub(" ", _ADMIN.sub(" ", norm))
    return not [w for w in rest.split() if w not in _FILLER and len(w) > 2]


def load_xy(
    name: str,
    *,
    drop_excluded: bool = True,
    drop_uninformative: bool = False,
    normalize_text: bool = False,
) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for row in load_split(name):
        code = row["code"]
        if drop_excluded and is_dropped(code):
            continue
        text = row["symptoms"]
        if drop_uninformative and is_uninformative(text):
            continue
        texts.append(normalize(text) if normalize_text else text)
        labels.append(group_of(code))
    return texts, labels
