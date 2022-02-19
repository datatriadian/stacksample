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
    answers_pickle = (
        Path(str(answers_file).replace(".csv", ".pkl"))
        if isinstance(answers_file, Path)
        else Path(answers_file.replace(".csv", ".pkl"))
    )
    if answers_pickle.exists():
        return pd.read_pickle(answers_pickle)

    df = pd.read_csv(answers_file, encoding=encoding)
    pd.to_pickle(df, answers_pickle)
    return df


@lru_cache(maxsize=1)
def load_questions(questions_file: Path | str, encoding: str) -> pd.DateFrame:
    questions_pickle = (
        Path(str(questions_file).replace(".csv", ".pkl"))
        if isinstance(questions_file, Path)
        else Path(questions_file.replace(".csv", ".pkl"))
    )
    if questions_pickle.exists():
        return pd.read_pickle(questions_pickle)

    df = pd.read_csv(questions_file, encoding=encoding)
    pd.to_pickle(df, questions_pickle)
    return df


@lru_cache(maxsize=1)
def load_tags(tags_file: Path | str, encoding: str) -> pd.DataFrame:
    tags_pickle = (
        Path(str(tags_file).replace(".csv", ".pkl"))
        if isinstance(tags_file, Path)
        else Path(tags_file.replace(".csv", ".pkl"))
    )
    if tags_pickle.exists():
        return pd.read_pickle(tags_pickle)

    df = pd.read_csv(tags_file, encoding=encoding)
    pd.to_pickle(df, tags_pickle)
    return df
