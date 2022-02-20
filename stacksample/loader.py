from __future__ import annotations

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


def load_answers(answers_file: Path | str, encoding: str) -> pd.DataFrame:
    return _loader(answers_file, encoding)


def load_questions(questions_file: Path | str, encoding: str) -> pd.DateFrame:
    return _loader(questions_file, encoding)


def load_tags(tags_file: Path | str, encoding: str) -> pd.DataFrame:
    return _loader(tags_file, encoding)


def _loader(file_path: Path | str, encoding: str) -> pd.DataFrame:
    file_pickle = (
        Path(str(file_path).replace(".csv", ".pkl"))
        if isinstance(file_path, Path)
        else Path(file_path.replace(".csv", ".pkl"))
    )
    if file_pickle.exists():
        return pd.read_pickle(file_pickle)

    df = pd.read_csv(file_path, encoding=encoding)
    pd.to_pickle(df, file_pickle)
    return df
