from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache(maxsize=1)
def load_answers(answers_file: Path | str = Path("data/Answers.csv")) -> pd.DataFrame:
    return pd.read_csv(answers_file, encoding="ISO-8859-1")


@lru_cache(maxsize=1)
def load_questions(questions_file: Path | str = Path("data/Questions.csv")) -> pd.DateOffset:
    return pd.read_csv(questions_file, encoding="ISO-8859-1")


@lru_cache(maxsize=1)
def load_tags(tags_file: Path | str = Path("data/Tags.csv")) -> pd.DataFrame:
    return pd.read_csv(tags_file, encoding="ISO-8859-1")
