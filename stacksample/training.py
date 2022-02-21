from __future__ import annotations

import json
import pickle
from multiprocessing import cpu_count
from pathlib import Path

import nltk
import pandas as pd
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB

from stacksample.console import console

nltk.download("stopwords")


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
    lowercase: bool = False,
    remove_stopwords: bool = False,
) -> pd.DataFrame:
    CACHE_PATH = Path("data/cache")
    if not CACHE_PATH.exists():
        CACHE_PATH.mkdir(parents=True, exist_ok=True)

    data_pickle = CACHE_PATH / "formatted.pkl"
    params_file = CACHE_PATH / "params.json"
    cache_params = {
        "crop_sentences": crop_sentences,
        "remove_html_tags": remove_html_tags,
        "remove_line_breaks": remove_line_breaks,
        "minimum_labels": minimum_labels,
        "reduce_number_of_samples": reduce_number_of_samples,
        "exclude_answers": exclude_answers,
        "exclude_title": exclude_title,
        "limit_tags": limit_tags,
        "random_state": random_state,
        "lowercase": lowercase,
    }

    if params_file.exists() and data_pickle.exists():
        with open(params_file, "r") as f:
            params = json.load(f)

        if params == cache_params:
            return pd.read_pickle(data_pickle)

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
    tags_df = _create_composit_labels(tags_df)

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

    if remove_stopwords:
        df = _remove_stopwords(df)

    if remove_line_breaks:
        df = _remove_line_breaks(df)

    if lowercase:
        df["sentences"] = df["sentences"].str.lower()

    with open(params_file, "w") as f:
        json.dump(cache_params, f)

    pd.to_pickle(df, data_pickle)
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
        X_train, y_train, X_test, y_test, vectorizer = _split_and_vectorize(
            df, test_size, random_state
        )

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
                save_classifier = save_path / "naive_bayes_classifier.pkl"
                save_vectorizer = save_path / "vectorizer.pkl"
            else:
                save_classifier = Path("./backend/models/naive_bays_classifier.pkl")
                save_vectorizer = Path("./backend/models/vectorizer.pkl")

            with open(save_classifier, "wb") as f:
                pickle.dump(gnb, f)

            with open(save_vectorizer, "wb") as f:
                pickle.dump(vectorizer, f)


def train_random_forest_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int | None = None,
    balance_train_dataset: bool = False,
    save_model: bool = True,
    save_path: Path | None = None,
    n_estimators: int = 100,
    n_jobs: int = cpu_count(),
) -> None:
    with console.status("Training Random Forest Model..."):
        clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
        X_train, y_train, X_test, y_test, vectorizer = _split_and_vectorize(
            df, test_size, random_state
        )

        console.print("Labels used for SVM training: ", y_train.unique())

        if balance_train_dataset:
            oversample = SMOTE(random_state=random_state)
            X_train, y_train = oversample.fit_resample(X_train, y_train)

        clf.fit(X_train, y_train)

    with console.status("Calculating Random Forest Accuracy..."):
        console.print(f"Random Forest Accuracy: {clf.score(X_test, y_test) * 100}%")

    with console.status("Calculating Random Forest f1 Scores..."):
        console.print(
            f"Random Forest f1 score: {f1_score(y_test, clf.predict(X_test), average=None)}"
        )

    if save_model:
        with console.status("Saving Random Forest model..."):
            if save_path:
                save_classifier = save_path / "random_forest_classifier.pkl"
                save_vectorizer = save_path / "vectorizer.pkl"
            else:
                save_classifier = Path("./backend/models/random_forest_classifier.pkl")
                save_vectorizer = Path("./backend/models/vectorizer.pkl")

            with open(save_classifier, "wb") as f:
                pickle.dump(clf, f)

            with open(save_vectorizer, "wb") as f:
                pickle.dump(vectorizer, f)


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
        X_train, y_train, X_test, y_test, vectorizer = _split_and_vectorize(
            df, test_size, random_state
        )

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
                save_classifier = save_path / "svm_classifier.pkl"
                save_vectorizer = save_path / "vectorizer.pkl"
            else:
                save_classifier = Path("./backend/models/svm_classifier.pkl")
                save_vectorizer = Path("./backend/models/vectorizer.pkl")

            with open(save_classifier, "wb") as f:
                pickle.dump(clf, f)

            with open(save_vectorizer, "wb") as f:
                pickle.dump(vectorizer, f)


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
        X_train, y_train, X_test, y_test, vectorizer = _split_and_vectorize(
            df, test_size, random_state
        )

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
                save_classifier = save_path / "svm_grid_search_classifier.pkl"
                save_vectorizer = save_path / "vectorizer.pkl"
            else:
                save_classifier = Path("./backend/models/svm_grid_search_classifier.pkl")
                save_vectorizer = Path("./backend/models/vectorizer.pkl")

            with open(save_classifier, "wb") as f:
                pickle.dump(clf, f)

            with open(save_vectorizer, "wb") as f:
                pickle.dump(vectorizer, f)


def _create_composit_labels(df: pd.DataFrame) -> pd.DataFrame:
    """There can be multiple tags for the same answer/question so combine these into one."""
    return df.dropna().sort_values(by=["tag"]).groupby("id")["tag"].apply(", ".join).reset_index()


def _crop_sentences(df: pd.DataFrame, max_length: int = 128) -> pd.DataFrame:
    df["sentences"] = df["sentences"].apply(lambda x: " ".join(x.split(" ")[:max_length]))
    return df


def _remove_html(df: pd.DataFrame) -> pd.DataFrame:
    df["sentences"] = df["sentences"].str.replace(r"<[^<>]*>", "", regex=True)
    return df


def _remove_line_breaks(df: pd.DataFrame) -> pd.DataFrame:
    df["sentences"] = df["sentences"].str.replace(r"\\n", " ", regex=True)
    return df


def _remove_stopwords(df: pd.DataFrame) -> pd.DataFrame:
    df["sentences"] = df["sentences"].apply(
        lambda x: " ".join([y for y in x.split(" ") if y.lower() not in stopwords.words("english")])
    )

    return df


def _split_and_vectorize(
    df: pd.DataFrame, test_size: float, random_state: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, TfidfVectorizer]:
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(df["sentences"])

    X_train = vectorizer.transform(train["sentences"])
    y_train = train["tag"]
    X_test = vectorizer.transform(test["sentences"])
    y_test = test["tag"]

    return X_train, y_train, X_test, y_test, vectorizer
