import pytest

from src.data import is_uninformative, normalize

ADMIN = [
    "Жалоб нет",
    "жалоб нет.",
    "Жалобы прежние",
    "Активно не предъявляет",
    "Повторный прием",
    "Плановая консультация",
    "Состояние без динамики",
]

COMPLAINTS = ["головокружение", "слабость", "боль в пояснице", "изжога после еды"]


@pytest.mark.parametrize("text", ADMIN)
def test_admin_records_dropped(text):
    assert is_uninformative(text)


@pytest.mark.parametrize("text", COMPLAINTS)
def test_short_complaints_kept(text):
    assert not is_uninformative(text)


def test_symptom_next_to_admin_phrase_kept():
    assert not is_uninformative("Жалоб нет, беспокоит кашель")


def test_normalize_collapses_whitespace():
    assert normalize("  БОЛЬ   в   спине \n") == "боль в спине"
