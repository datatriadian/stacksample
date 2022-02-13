from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

_MODEL_PATH = Path().absolute() / "models"


@lru_cache(maxsize=1)
def load_classifier() -> SVC:
    with open(_MODEL_PATH / "svm_classifier.pkl", "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_vectorizer() -> TfidfVectorizer:
    with open(_MODEL_PATH / "vectorizer.pkl", "rb") as f:
        return pickle.load(f)
