from __future__ import annotations

import pickle
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB

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
    limit_tags: int | None = None,
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

    df = df.merge(tags_df, on="id")
    df = df[["sentences", "tag"]]

    if reduce_number_of_samples:
        df = df.sample(n=reduce_number_of_samples, random_state=random_state)

    if minimum_labels:
        keep = df[df.groupby("tag")["tag"].transform("size") >= minimum_labels]
        df = df[df["tag"].isin(keep["tag"])]

    if limit_tags:
        limit_df = (
            df.groupby("tag")
            .size()
            .reset_index(name="count")
            .sort_values(["count"], ascending=False)
            .head(limit_tags)
        )

        df = df[df["tag"].isin(limit_df["tag"])]

    if remove_html_tags:
        df = _remove_html(df)

    if crop_sentences:
        df = _crop_sentences(df, crop_sentences)

    if remove_line_breaks:
        df = _remove_line_breaks(df)

    return df


def train_naive_bayes_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int | None = None,
    balance_train_dataset: bool = False,
    save_model: bool = True,
    save_path: Path | None = None,
) -> None:
    with console.status("Training Naive Bayes Model..."):
        gnb = GaussianNB()
        X_train, y_train, X_test, y_test = _split_and_vectorize(df, test_size, random_state)

        console.print("Labels used for Naive Bayes training: ", y_train.unique())

        if balance_train_dataset:
            oversample = SMOTE(random_state=random_state)
            X_train, y_train = oversample.fit_resample(X_train, y_train)

        gnb.fit(X_train.todense(), y_train)

    with console.status("Calculating Naive Bayes Accuracy..."):
        console.print(f"Naive Bayes Accuracy: {gnb.score(X_test.todense(), y_test) * 100}%")

    with console.status("Calculating Naive Bayes f1 Scores..."):
        console.print(
            f"Naive Bays f1 score: {f1_score(y_test, gnb.predict(X_test.todense()), average=None)}"
        )

    if save_model:
        with console.status("Saving Naive Bayes model..."):
            if save_path:
                save = save_path
            else:
                save = Path("./models/naive_bayes_classifier.plk")

            with open(save, "wb") as f:
                pickle.dump(gnb, f)


def train_svm_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int | None = None,
    balance_train_dataset: bool = False,
    c_value: float = 1.0,
    save_model: bool = True,
    save_path: Path | None = None,
) -> None:
    with console.status("Training SVM Model..."):
        clf = svm.SVC(C=c_value, kernel="linear", gamma="auto")
        X_train, y_train, X_test, y_test = _split_and_vectorize(df, test_size, random_state)

        console.print("Labels used for SVM training: ", y_train.unique())

        if balance_train_dataset:
            oversample = SMOTE(random_state=random_state)
            X_train, y_train = oversample.fit_resample(X_train, y_train)

        clf.fit(X_train, y_train)

    with console.status("Calculating SVM Accuracy..."):
        console.print(f"SVM Accuracy: {clf.score(X_test, y_test) * 100}%")

    with console.status("Calculating SVM f1 Scores..."):
        console.print(f"SVM f1 score: {f1_score(y_test, clf.predict(X_test), average=None)}")

    if save_model:
        with console.status("Saving SVM model..."):
            if save_path:
                save = save_path
            else:
                save = Path("./models/svm_classifier.plk")

            with open(save, "wb") as f:
                pickle.dump(clf, f)


def train_svm_model_grid_search(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int | None = None,
    balance_train_dataset: bool = False,
    n_jobs: int = cpu_count(),
    save_model: bool = True,
    save_path: Path | None = None,
) -> None:
    with console.status("Training SVM Model With Grid Search..."):
        parameters = {
            "kernel": ("linear", "rbf"),
            "C": (0.1, 1, 8, 16, 32),
            "gamma": ("scale", "auto"),
        }
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters, n_jobs=n_jobs)
        X_train, y_train, X_test, y_test = _split_and_vectorize(df, test_size, random_state)

        console.print("Labels used for SVM training: ", y_train.unique())

        if balance_train_dataset:
            oversample = SMOTE(random_state=random_state)
            X_train, y_train = oversample.fit_resample(X_train, y_train)

        clf.fit(X_train, y_train)
        console.print(f"Best parameters: {clf.best_params_}")

    with console.status("Calculating SVM Accuracy..."):
        console.print(f"SVM Accuracy: {clf.score(X_test, y_test) * 100}%")

    with console.status("Calculating SVM f1 Scores..."):
        console.print(f"SVM f1 score: {f1_score(y_test, clf.predict(X_test), average=None)}")

    if save_model:
        with console.status("Saving SVM grid search model..."):
            if save_path:
                save = save_path
            else:
                save = Path("./models/svm_grid_search_classifier.plk")

            with open(save, "wb") as f:
                pickle.dump(clf, f)


def _crop_sentences(df: pd.DataFrame, max_length: int = 128) -> pd.DataFrame:
    df["sentences"] = df["sentences"].apply(lambda x: " ".join(x.split(" ")[:max_length]))
    return df


def _remove_html(df: pd.DataFrame) -> pd.DataFrame:
    df["sentences"] = df["sentences"].str.replace(r"<[^<>]*>", "", regex=True)
    return df


def _remove_line_breaks(df: pd.DataFrame) -> pd.DataFrame:
    df["sentences"] = df["sentences"].str.replace(r"\\n", " ", regex=True)
    return df


def _split_and_vectorize(
    df: pd.DataFrame, test_size: float, random_state: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(df["sentences"])

    X_train = vectorizer.transform(train["sentences"])
    y_train = train["tag"]
    X_test = vectorizer.transform(test["sentences"])
    y_test = test["tag"]

    return X_train, y_train, X_test, y_test
