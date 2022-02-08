from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


def load_all(
    *,
    answers_file: Path | str,
    answers_encoding: str,
    questions_file: Path | str,
    questions_encoding: str,
    tags_file: Path | str,
    tags_encoding: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    answers = load_answers(answers_file, answers_encoding)
    questions = load_questions(questions_file, questions_encoding)
    tags = load_tags(tags_file, tags_encoding)

    return answers, questions, tags


@lru_cache(maxsize=1)
def load_answers(answers_file: Path | str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(answers_file, encoding=encoding)


@lru_cache(maxsize=1)
def load_questions(questions_file: Path | str, encoding: str) -> pd.DateFrame:
    return pd.read_csv(questions_file, encoding=encoding)


@lru_cache(maxsize=1)
def load_tags(tags_file: Path | str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(tags_file, encoding=encoding)
