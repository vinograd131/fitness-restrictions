import pytest

from src.config import DATA_DIR


def pytest_collection_modifyitems(config, items):
    if (DATA_DIR / "train_v1.jsonl").exists():
        return
    skip = pytest.mark.skip(reason="нет data/train_v1.jsonl, см. data/README.md")
    for item in items:
        item.add_marker(skip)
