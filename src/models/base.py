"""Общий интерфейс моделей: оценка и сервис работают с любой из них одинаково."""
from abc import ABC, abstractmethod

import numpy as np


class BaseClassifier(ABC):
    name: str

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """[n_texts, n_classes] в порядке classes."""

    @property
    @abstractmethod
    def classes(self) -> list[str]:
        ...

    def predict(self, texts: list[str]) -> list[str]:
        proba = self.predict_proba(texts)
        return [self.classes[i] for i in proba.argmax(1)]
