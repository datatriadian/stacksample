from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional

import pandas as pd
from typer import Option, Typer

from stacksample.console import console
from stacksample.loader import load_all
from stacksample.training import (
    combine_and_format_data,
    train_naive_bayes_model,
    train_svm_model,
    train_svm_model_grid_search,
)

app = Typer()
_DEFAULT_ENCODING = "ISO8859-1"


@app.command()
def view_labels(
    answers_file_path: Path = Option(
        Path("data/Answers.csv"), help="The path to the answers file. Default = data/Answers.csv"
    ),
    answers_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the answers file. Default = {_DEFAULT_ENCODING}"
    ),
    questions_file_path: Path = Option(
        Path("data/Questions.csv"),
        help="The path to the quesions file. Default = data/Questions.csv",
    ),
    questions_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the questions file. Default = {_DEFAULT_ENCODING}"
    ),
    tags_file_path: Path = Option(
        Path("data/Tags.csv"), help="The path to the tags file. Default = data/Tags.csv"
    ),
    tags_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the tags file. Default = {_DEFAULT_ENCODING}"
    ),
    remove_html_tags: bool = Option(
        False, help="If this flag is set HTML tags will be removed. Default = False"
    ),
    remove_line_breaks: bool = Option(
        False, help="If this flag is set line breaks will be removed. Default = False"
    ),
    minimum_labels: Optional[int] = Option(
        None, help="Only keep labels with the specified number of samples. Default = None"
    ),
    reduce_number_of_samples: Optional[int] = Option(
        None,
        help="If set this will be the maximum number of records used for testing and training. Default = None",
    ),
    random_state: Optional[int] = Option(
        None, help="Random state to use for reproducability. Default = None"
    ),
    crop_sentences: Optional[int] = Option(
        None, help="Crop sentences to a maximum number of characters. Default = None"
    ),
    exclude_title: bool = Option(
        False, help="If this flag is set the title column in will be excluded. Default = False"
    ),
    exclude_answers: bool = Option(
        False, help="If this flag is set the answers will be excluded. Default = False"
    ),
    limit_tags: Optional[int] = Option(
        None, help="Specifies the maximum number of tags to use in training. Default = None"
    ),
) -> None:
    with console.status("Loading data..."):
        answers, questions, tags = load_all(
            answers_file=answers_file_path,
            answers_encoding=answers_file_encoding,
            questions_file=questions_file_path,
            questions_encoding=questions_file_encoding,
            tags_file=tags_file_path,
            tags_encoding=tags_file_encoding,
        )

    with console.status("Preparing data..."):
        df = combine_and_format_data(
            answers=answers,
            questions=questions,
            tags=tags,
            remove_html_tags=remove_html_tags,
            remove_line_breaks=remove_line_breaks,
            minimum_labels=minimum_labels,
            reduce_number_of_samples=reduce_number_of_samples,
            crop_sentences=crop_sentences,
            random_state=random_state,
            exclude_answers=exclude_answers,
            exclude_title=exclude_title,
            limit_tags=limit_tags,
        )

    pd.set_option("display.max_rows", df.shape[0] + 1)
    console.print(df.groupby("tag").size())


@app.command()
def train_all_models(
    answers_file_path: Path = Option(
        Path("data/Answers.csv"), help="The path to the answers file. Default = data/Answers.csv"
    ),
    answers_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the answers file. Default = {_DEFAULT_ENCODING}"
    ),
    questions_file_path: Path = Option(
        Path("data/Questions.csv"),
        help="The path to the quesions file. Default = data/Questions.csv",
    ),
    questions_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the questions file. Default = {_DEFAULT_ENCODING}"
    ),
    tags_file_path: Path = Option(
        Path("data/Tags.csv"), help="The path to the tags file. Default = data/Tags.csv"
    ),
    tags_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the tags file. Default = {_DEFAULT_ENCODING}"
    ),
    remove_html_tags: bool = Option(
        False, help="If this flag is set HTML tags will be removed. Default = False"
    ),
    remove_line_breaks: bool = Option(
        False, help="If this flag is set line breaks will be removed. Default = False"
    ),
    reduce_number_of_samples: Optional[int] = Option(
        None,
        help="If set this will be the maximum number of records used for testing and training. Default = None",
    ),
    test_size: float = Option(
        0.2,
        help="The percentage of data to be used for training as a float. Default = 0.2",
    ),
    random_state: Optional[int] = Option(
        None, help="Random state to use for reproducability. Default = None"
    ),
    minimum_labels: Optional[int] = Option(
        None, help="Only keep labels with the specified number of samples. Default = None"
    ),
    crop_sentences: Optional[int] = Option(
        None, help="Crop sentences to a maximum number of characters. Default = None"
    ),
    exclude_title: bool = Option(
        False, help="If this flag is set the title column in will be excluded. Default = False"
    ),
    exclude_answers: bool = Option(
        False, help="If this flag is set the answers will be excluded. Default = False"
    ),
    limit_tags: Optional[int] = Option(
        None, help="Specifies the maximum number of tags to use in training. Default = None"
    ),
    balance_train_data: bool = Option(
        False,
        help="If this flag is set oversampling will be preformed on the train data. Default = False",
    ),
    c_value: float = Option(1.0, "-c", help="Sets the C value for the SVM. Default = 1.0"),
    save_model: bool = Option(True, help="Save the model after training. Default = True"),
    naive_bayes_save_path: Optional[Path] = Option(
        None,
        help="The path and file name for saving the naive bayes model if saving the model. Default = None",
    ),
    svm_save_path: Optional[Path] = Option(
        None,
        help="The path and file name for saving the SVM model if saving the model. Default = None",
    ),
) -> None:
    answers, questions, tags = _load_all(
        answers_file_path=answers_file_path,
        answers_file_encoding=answers_file_encoding,
        questions_file_path=questions_file_path,
        questions_file_encoding=questions_file_encoding,
        tags_file_path=tags_file_path,
        tags_file_encoding=tags_file_encoding,
    )

    df = _combine_and_format_data(
        answers=answers,
        questions=questions,
        tags=tags,
        remove_html_tags=remove_html_tags,
        remove_line_breaks=remove_line_breaks,
        minimum_labels=minimum_labels,
        reduce_number_of_samples=reduce_number_of_samples,
        crop_sentences=crop_sentences,
        random_state=random_state,
        exclude_answers=exclude_answers,
        exclude_title=exclude_title,
        limit_tags=limit_tags,
    )

    train_naive_bayes_model(
        df,
        test_size=test_size,
        random_state=random_state,
        balance_train_dataset=balance_train_data,
        save_model=save_model,
        save_path=naive_bayes_save_path,
    )

    train_svm_model(
        df,
        test_size=test_size,
        random_state=random_state,
        balance_train_dataset=balance_train_data,
        c_value=c_value,
        save_model=save_model,
        save_path=svm_save_path,
    )


@app.command()
def train_naive_bayes(
    answers_file_path: Path = Option(
        Path("data/Answers.csv"), help="The path to the answers file. Default = data/Answers.csv"
    ),
    answers_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the answers file. Default = {_DEFAULT_ENCODING}"
    ),
    questions_file_path: Path = Option(
        Path("data/Questions.csv"),
        help="The path to the quesions file. Default = data/Questions.csv",
    ),
    questions_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the questions file. Default = {_DEFAULT_ENCODING}"
    ),
    tags_file_path: Path = Option(
        Path("data/Tags.csv"), help="The path to the tags file. Default = data/Tags.csv"
    ),
    tags_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the tags file. Default = {_DEFAULT_ENCODING}"
    ),
    remove_html_tags: bool = Option(
        False, help="If this flag is set HTML tags will be removed. Default = False"
    ),
    remove_line_breaks: bool = Option(
        False, help="If this flag is set line breaks will be removed. Default = False"
    ),
    reduce_number_of_samples: Optional[int] = Option(
        None,
        help="If set this will be the maximum number of records used for testing and training. Default = None",
    ),
    test_size: float = Option(
        0.2,
        help="The percentage of data to be used for training as a float. Default = 0.2",
    ),
    random_state: Optional[int] = Option(
        None, help="Random state to use for reproducability. Default = None"
    ),
    minimum_labels: Optional[int] = Option(
        None, help="Only keep labels with the specified number of samples. Default = None"
    ),
    crop_sentences: Optional[int] = Option(
        None, help="Crop sentences to a maximum number of characters. Default = None"
    ),
    exclude_title: bool = Option(
        False, help="If this flag is set the title column in will be excluded. Default = False"
    ),
    exclude_answers: bool = Option(
        False, help="If this flag is set the answers will be excluded. Default = False"
    ),
    limit_tags: Optional[int] = Option(
        None, help="Specifies the maximum number of tags to use in training. Default = None"
    ),
    balance_train_data: bool = Option(
        False,
        help="If this flag is set oversampling will be preformed on the train data. Default = False",
    ),
    save_model: bool = Option(True, help="Save the model after training. Default = True"),
    save_path: Optional[Path] = Option(
        None, help="The path and file name for saving the model if saving the model. Default = None"
    ),
) -> None:
    answers, questions, tags = _load_all(
        answers_file_path=answers_file_path,
        answers_file_encoding=answers_file_encoding,
        questions_file_path=questions_file_path,
        questions_file_encoding=questions_file_encoding,
        tags_file_path=tags_file_path,
        tags_file_encoding=tags_file_encoding,
    )

    df = _combine_and_format_data(
        answers=answers,
        questions=questions,
        tags=tags,
        remove_html_tags=remove_html_tags,
        remove_line_breaks=remove_line_breaks,
        minimum_labels=minimum_labels,
        reduce_number_of_samples=reduce_number_of_samples,
        crop_sentences=crop_sentences,
        random_state=random_state,
        exclude_answers=exclude_answers,
        exclude_title=exclude_title,
        limit_tags=limit_tags,
    )

    train_naive_bayes_model(
        df,
        test_size=test_size,
        random_state=random_state,
        balance_train_dataset=balance_train_data,
        save_model=save_model,
        save_path=save_path,
    )


@app.command()
def train_svm(
    answers_file_path: Path = Option(
        Path("data/Answers.csv"), help="The path to the answers file. Default = data/Answers.csv"
    ),
    answers_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the answers file. Default = {_DEFAULT_ENCODING}"
    ),
    questions_file_path: Path = Option(
        Path("data/Questions.csv"),
        help="The path to the quesions file. Default = data/Questions.csv",
    ),
    questions_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the questions file. Default = {_DEFAULT_ENCODING}"
    ),
    tags_file_path: Path = Option(
        Path("data/Tags.csv"), help="The path to the tags file. Default = data/Tags.csv"
    ),
    tags_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the tags file. Default = {_DEFAULT_ENCODING}"
    ),
    remove_html_tags: bool = Option(
        False, help="If this flag is set HTML tags will be removed. Default = False"
    ),
    remove_line_breaks: bool = Option(
        False, help="If this flag is set line breaks will be removed. Default = False"
    ),
    reduce_number_of_samples: Optional[int] = Option(
        None,
        help="If set this will be the maximum number of records used for testing and training. Default = None",
    ),
    test_size: float = Option(
        0.2,
        help="The percentage of data to be used for training as a float. Default = 0.2",
    ),
    random_state: Optional[int] = Option(
        None, help="Random state to use for reproducability. Default = None"
    ),
    minimum_labels: Optional[int] = Option(
        None, help="Only keep labels with the specified number of samples. Default = None"
    ),
    crop_sentences: Optional[int] = Option(
        None, help="Crop sentences to a maximum number of characters. Default = None"
    ),
    exclude_title: bool = Option(
        False, help="If this flag is set the title column in will be excluded. Default = False"
    ),
    exclude_answers: bool = Option(
        False, help="If this flag is set the answers will be excluded. Default = False"
    ),
    limit_tags: Optional[int] = Option(
        None, help="Specifies the maximum number of tags to use in training. Default = None"
    ),
    balance_train_data: bool = Option(
        False,
        help="If this flag is set oversampling will be preformed on the train data. Default = False",
    ),
    c_value: float = Option(1.0, "-c", help="Sets the C value for the SVM. Default = 1.0"),
    save_model: bool = Option(True, help="Save the model after training. Default = True"),
    save_path: Optional[Path] = Option(
        None, help="The path and file name for saving the model if saving the model. Default = None"
    ),
) -> None:
    answers, questions, tags = _load_all(
        answers_file_path=answers_file_path,
        answers_file_encoding=answers_file_encoding,
        questions_file_path=questions_file_path,
        questions_file_encoding=questions_file_encoding,
        tags_file_path=tags_file_path,
        tags_file_encoding=tags_file_encoding,
    )

    df = _combine_and_format_data(
        answers=answers,
        questions=questions,
        tags=tags,
        remove_html_tags=remove_html_tags,
        remove_line_breaks=remove_line_breaks,
        minimum_labels=minimum_labels,
        reduce_number_of_samples=reduce_number_of_samples,
        crop_sentences=crop_sentences,
        random_state=random_state,
        exclude_answers=exclude_answers,
        exclude_title=exclude_title,
        limit_tags=limit_tags,
    )

    train_svm_model(
        df,
        test_size=test_size,
        random_state=random_state,
        balance_train_dataset=balance_train_data,
        c_value=c_value,
        save_model=save_model,
        save_path=save_path,
    )


@app.command()
def train_svm_grid_search(
    answers_file_path: Path = Option(
        Path("data/Answers.csv"), help="The path to the answers file. Default = data/Answers.csv"
    ),
    answers_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the answers file. Default = {_DEFAULT_ENCODING}"
    ),
    questions_file_path: Path = Option(
        Path("data/Questions.csv"),
        help="The path to the quesions file. Default = data/Questions.csv",
    ),
    questions_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the questions file. Default = {_DEFAULT_ENCODING}"
    ),
    tags_file_path: Path = Option(
        Path("data/Tags.csv"), help="The path to the tags file. Default = data/Tags.csv"
    ),
    tags_file_encoding: str = Option(
        _DEFAULT_ENCODING, help=f"The encoding of the tags file. Default = {_DEFAULT_ENCODING}"
    ),
    remove_html_tags: bool = Option(
        False, help="If this flag is set HTML tags will be removed. Default = False"
    ),
    remove_line_breaks: bool = Option(
        False, help="If this flag is set line breaks will be removed. Default = False"
    ),
    reduce_number_of_samples: Optional[int] = Option(
        None,
        help="If set this will be the maximum number of records used for testing and training. Default = None",
    ),
    test_size: float = Option(
        0.2,
        help="The percentage of data to be used for training as a float. Default = 0.2",
    ),
    random_state: Optional[int] = Option(
        None, help="Random state to use for reproducability. Default = None"
    ),
    minimum_labels: Optional[int] = Option(
        None, help="Only keep labels with the specified number of samples. Default = None"
    ),
    crop_sentences: Optional[int] = Option(
        None, help="Crop sentences to a maximum number of characters. Default = None"
    ),
    exclude_title: bool = Option(
        False, help="If this flag is set the title column in will be excluded. Default = False"
    ),
    exclude_answers: bool = Option(
        False, help="If this flag is set the answers will be excluded. Default = False"
    ),
    limit_tags: Optional[int] = Option(
        None, help="Specifies the maximum number of tags to use in training. Default = None"
    ),
    balance_train_data: bool = Option(
        False,
        help="If this flag is set oversampling will be preformed on the train data. Default = False",
    ),
    n_jobs: int = Option(
        cpu_count(),
        help="The number of CPU cores to use for training. Default = The number of available CPU cores",
    ),
    save_model: bool = Option(True, help="Save the model after training. Default = True"),
    save_path: Optional[Path] = Option(
        None, help="The path and file name for saving the model if saving the model. Default = None"
    ),
) -> None:
    answers, questions, tags = _load_all(
        answers_file_path=answers_file_path,
        answers_file_encoding=answers_file_encoding,
        questions_file_path=questions_file_path,
        questions_file_encoding=questions_file_encoding,
        tags_file_path=tags_file_path,
        tags_file_encoding=tags_file_encoding,
    )

    df = _combine_and_format_data(
        answers=answers,
        questions=questions,
        tags=tags,
        remove_html_tags=remove_html_tags,
        remove_line_breaks=remove_line_breaks,
        minimum_labels=minimum_labels,
        reduce_number_of_samples=reduce_number_of_samples,
        crop_sentences=crop_sentences,
        random_state=random_state,
        exclude_answers=exclude_answers,
        exclude_title=exclude_title,
        limit_tags=limit_tags,
    )

    train_svm_model_grid_search(
        df,
        test_size=test_size,
        random_state=random_state,
        balance_train_dataset=balance_train_data,
        n_jobs=n_jobs,
        save_model=save_model,
        save_path=save_path,
    )


def _load_all(
    answers_file_path: Path,
    answers_file_encoding: str,
    questions_file_path: Path,
    questions_file_encoding: str,
    tags_file_path: Path,
    tags_file_encoding: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with console.status("Loading data..."):
        answers, questions, tags = load_all(
            answers_file=answers_file_path,
            answers_encoding=answers_file_encoding,
            questions_file=questions_file_path,
            questions_encoding=questions_file_encoding,
            tags_file=tags_file_path,
            tags_encoding=tags_file_encoding,
        )

    return answers, questions, tags


def _combine_and_format_data(
    answers: pd.DataFrame,
    questions: pd.DataFrame,
    tags: pd.DataFrame,
    remove_html_tags: bool,
    remove_line_breaks: bool,
    minimum_labels: int | None,
    reduce_number_of_samples: int | None,
    crop_sentences: int | None,
    random_state: int | None,
    exclude_answers: bool,
    exclude_title: bool,
    limit_tags: int | None,
) -> pd.DataFrame:
    with console.status("Preparing data..."):
        df = combine_and_format_data(
            answers=answers,
            questions=questions,
            tags=tags,
            remove_html_tags=remove_html_tags,
            remove_line_breaks=remove_line_breaks,
            minimum_labels=minimum_labels,
            reduce_number_of_samples=reduce_number_of_samples,
            crop_sentences=crop_sentences,
            random_state=random_state,
            exclude_answers=exclude_answers,
            exclude_title=exclude_title,
            limit_tags=limit_tags,
        )

    return df


if __name__ == "__main__":
    app()
