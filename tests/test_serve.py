"""Smoke-тесты REST-сервиса на лёгкой baseline-модели."""
import os

os.environ["MODEL"] = "baseline"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.config import threshold_for  # noqa: E402
from src.mapping import GROUPS, RESTRICTIONS  # noqa: E402
from src.serve import app  # noqa: E402

pytestmark = pytest.mark.needs_data


def test_ping():
    with TestClient(app) as client:
        assert client.get("/ping").json()["status"] == "ok"


def test_groups_cover_all():
    with TestClient(app) as client:
        body = client.get("/groups").json()
    assert len(body) == len(GROUPS)
    assert {row["group"] for row in body} == set(GROUPS)


def test_predict_shape():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"text": "изжога и тяжесть в животе после еды"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["group"] in GROUPS
    rec = RESTRICTIONS[body["group"]]
    assert body["summary"] == rec["summary"]
    assert body["forbidden"] == rec["forbidden"]
    assert body["equipment"] == rec["equipment"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_empty():
    with TestClient(app) as client:
        assert client.post("/predict", json={"text": ""}).status_code == 422


def test_low_confidence_asks_to_clarify():
    """Невнятная жалоба — сервис просит уточнить и отдаёт кандидатов."""
    with TestClient(app) as client:
        body = client.post("/predict", json={"text": "плохо себя чувствую"}).json()

    assert body["needs_clarification"] == (body["confidence"] < threshold_for("baseline"))
    if body["needs_clarification"]:
        assert body["message"]
        assert len(body["alternatives"]) > 1


def test_distribution_covers_all_groups():
    with TestClient(app) as client:
        body = client.post("/predict", json={"text": "болит колено"}).json()
    dist = body["distribution"]
    assert {d["group"] for d in dist} == set(GROUPS)
    assert abs(sum(d["confidence"] for d in dist) - 1) < 0.01
    assert dist == sorted(dist, key=lambda d: -d["confidence"])
