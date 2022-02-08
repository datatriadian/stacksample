from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from stacksample.console import console


def combine_and_format_data(
    *,
    answers: pd.DataFrame,
    questions: pd.DataFrame,
    tags: pd.DataFrame,
    crop_sentences: int | None = None,
    remove_html_tags: bool = False,
    remove_line_breaks: bool = False,
    minimum_labels: int | None = None,
    reduce_number_of_samples: int | None = None,
    exclude_answers: bool = False,
    exclude_title: bool = False,
    random_state: int | None = None,
) -> pd.DataFrame:
    df = questions[["Id", "Body"]].rename(columns={"Id": "id", "Body": "sentences"})

    if not exclude_title:
        questions_title_df = questions[["Id", "Title"]].rename(
            columns={"Id": "id", "Title": "sentences"}
        )
        df = pd.concat([df, questions_title_df])

    if not exclude_answers:
        answers_df = answers[["ParentId", "Body"]].rename(
            columns={"ParentId": "id", "Body": "sentences"}
        )
        df = pd.concat([df, answers_df])

    tags_df = tags.rename(columns={"Id": "id", "Tag": "tag"})

    # There can be multiple tags for the same answer/question so combine these into one
    tags_df = tags_df.dropna().groupby("id")["tag"].apply(", ".join).reset_index()

    df = df.merge(tags_df, how="left", on="id")
    df = df[["sentences", "tag"]]

    if reduce_number_of_samples:
        df = df.sample(n=reduce_number_of_samples, random_state=random_state)

    if minimum_labels:
        keep = df[df.groupby("tag")["tag"].transform("size") >= minimum_labels]
        df = df[df["tag"].isin(keep["tag"])]

    if crop_sentences:
        df = _crop_sentences(df, crop_sentences)

    if remove_html_tags:
        df = _remove_html(df)

    if remove_line_breaks:
        df = _remove_line_breaks(df)

    return df


def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int | None = None,
    balance_train_dataset: bool = False,
    c_value: float = 1.0,
) -> None:
    clf = svm.SVC(C=c_value, kernel="linear", gamma="auto")
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(df["sentences"])

    X_train = vectorizer.transform(train["sentences"])
    y_train = train["tag"]
    X_test = vectorizer.transform(test["sentences"])
    y_test = test["tag"]

    console.print("Labels used for training: ", y_train.unique())

    if balance_train_dataset:
        oversample = SMOTE(random_state=random_state)
        X_train, y_train = oversample.fit_resample(X_train, y_train)

    clf.fit(X_train, y_train)
    console.print(f"Accuracy: {clf.score(X_test, y_test) * 100}%")
    console.print(f"f1 score: {f1_score(y_test, clf.predict(X_test), average=None)}")


def _crop_sentences(df: pd.DataFrame, max_length: int = 128) -> pd.DataFrame:
    df["sentences"] = df["sentences"].apply(lambda x: " ".join(x.split(" ")[:max_length]))
    return df


def _remove_html(df: pd.DataFrame) -> pd.DataFrame:
    df["sentences"] = df["sentences"].str.replace(r"<[^<>]*>", "", regex=True)
    return df


def _remove_line_breaks(df: pd.DataFrame) -> pd.DataFrame:
    df["sentences"] = df["sentences"].str.replace(r"\\n", " ", regex=True)
    return df
