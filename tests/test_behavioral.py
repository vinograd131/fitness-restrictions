from collections import Counter

from src.evaluation.behavioral import LEXICON, MFT_CASES, TEMPLATES
from src.mapping import GROUPS


def test_lexicon_covers_all_groups():
    assert set(LEXICON) == set(GROUPS)


def test_equal_cases_per_group():
    counts = Counter(group for _, group in MFT_CASES)
    assert len(set(counts.values())) == 1, counts


def test_cases_are_unique():
    texts = [text for text, _ in MFT_CASES]
    assert len(texts) == len(set(texts))


def test_size_matches_lexicon():
    symptoms = sum(len(v) for v in LEXICON.values())
    assert len(MFT_CASES) == symptoms * len(TEMPLATES)
