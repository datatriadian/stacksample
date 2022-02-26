from stacksample.main import app


def test_train_all_models_no_save(mock_data_directory, test_runner, tmp_path):
    args = [
        "train-all-models",
        f"--answers-file-path={str(mock_data_directory / 'Answers.csv')}",
        "--answers-file-encoding=ISO8859-1",
        f"--questions-file-path={str(mock_data_directory / 'Questions.csv')}",
        "--questions-file-encoding=ISO8859-1",
        f"--tags-file-path={str(mock_data_directory / 'Tags.csv')}",
        "--tags-file-encoding=ISO8859-1",
        f"--naive-bayes-save-path={str(tmp_path)}",
        f"--svm-save-path={str(tmp_path)}",
        f"--random-forest-save-path={str(tmp_path)}",
        "--test-size=0.01",
    ]
    test_runner.invoke(app, args, catch_exceptions=False)

    assert len(list(tmp_path.glob("*.pkl"))) == 0


def test_train_all_models_save(mock_data_directory, test_runner, tmp_path):
    args = [
        "train-all-models",
        f"--answers-file-path={str(mock_data_directory / 'Answers.csv')}",
        "--answers-file-encoding=ISO8859-1",
        f"--questions-file-path={str(mock_data_directory / 'Questions.csv')}",
        "--questions-file-encoding=ISO8859-1",
        f"--tags-file-path={str(mock_data_directory / 'Tags.csv')}",
        "--tags-file-encoding=ISO8859-1",
        "--save-model",
        f"--naive-bayes-save-path={str(tmp_path)}",
        f"--svm-save-path={str(tmp_path)}",
        f"--random-forest-save-path={str(tmp_path)}",
        "--test-size=0.01",
    ]
    result = test_runner.invoke(app, args, catch_exceptions=False)
    result.stdout

    assert len(list(tmp_path.glob("*.pkl"))) == 4


def test_train_naive_bayes_no_save(mock_data_directory, test_runner, tmp_path):
    args = [
        "train-naive-bayes",
        f"--answers-file-path={str(mock_data_directory / 'Answers.csv')}",
        "--answers-file-encoding=ISO8859-1",
        f"--questions-file-path={str(mock_data_directory / 'Questions.csv')}",
        "--questions-file-encoding=ISO8859-1",
        f"--tags-file-path={str(mock_data_directory / 'Tags.csv')}",
        "--tags-file-encoding=ISO8859-1",
        f"--save-path={str(tmp_path)}",
        "--test-size=0.01",
    ]
    test_runner.invoke(app, args, catch_exceptions=False)

    assert len(list(tmp_path.glob("*.pkl"))) == 0


def test_train_naive_bayes_save(mock_data_directory, test_runner, tmp_path):
    args = [
        "train-naive-bayes",
        f"--answers-file-path={str(mock_data_directory / 'Answers.csv')}",
        "--answers-file-encoding=ISO8859-1",
        f"--questions-file-path={str(mock_data_directory / 'Questions.csv')}",
        "--questions-file-encoding=ISO8859-1",
        f"--tags-file-path={str(mock_data_directory / 'Tags.csv')}",
        "--tags-file-encoding=ISO8859-1",
        "--save-model",
        f"--save-path={str(tmp_path)}",
        "--test-size=0.01",
    ]
    test_runner.invoke(app, args, catch_exceptions=False)

    assert len(list(tmp_path.glob("*.pkl"))) == 2


def test_train_svm_no_save(mock_data_directory, test_runner, tmp_path):
    args = [
        "train-svm",
        f"--answers-file-path={str(mock_data_directory / 'Answers.csv')}",
        "--answers-file-encoding=ISO8859-1",
        f"--questions-file-path={str(mock_data_directory / 'Questions.csv')}",
        "--questions-file-encoding=ISO8859-1",
        f"--tags-file-path={str(mock_data_directory / 'Tags.csv')}",
        "--tags-file-encoding=ISO8859-1",
        f"--save-path={str(tmp_path)}",
        "--test-size=0.01",
    ]
    test_runner.invoke(app, args, catch_exceptions=False)

    assert len(list(tmp_path.glob("*.pkl"))) == 0


def test_train_svm_save(mock_data_directory, test_runner, tmp_path):
    args = [
        "train-svm",
        f"--answers-file-path={str(mock_data_directory / 'Answers.csv')}",
        "--answers-file-encoding=ISO8859-1",
        f"--questions-file-path={str(mock_data_directory / 'Questions.csv')}",
        "--questions-file-encoding=ISO8859-1",
        f"--tags-file-path={str(mock_data_directory / 'Tags.csv')}",
        "--tags-file-encoding=ISO8859-1",
        "--save-model",
        f"--save-path={str(tmp_path)}",
        "--test-size=0.01",
    ]
    test_runner.invoke(app, args, catch_exceptions=False)

    assert len(list(tmp_path.glob("*.pkl"))) == 2


def test_train_svm_grid_search_no_save(mock_data_directory, test_runner, tmp_path):
    args = [
        "train-svm-grid-search",
        f"--answers-file-path={str(mock_data_directory / 'Answers.csv')}",
        "--answers-file-encoding=ISO8859-1",
        f"--questions-file-path={str(mock_data_directory / 'Questions.csv')}",
        "--questions-file-encoding=ISO8859-1",
        f"--tags-file-path={str(mock_data_directory / 'Tags.csv')}",
        "--tags-file-encoding=ISO8859-1",
        f"--save-path={str(tmp_path)}",
        "--test-size=0.01",
    ]
    test_runner.invoke(app, args, catch_exceptions=False)

    assert len(list(tmp_path.glob("*.pkl"))) == 0


def test_train_svm_grid_search_save(mock_data_directory, test_runner, tmp_path):
    args = [
        "train-svm-grid-search",
        f"--answers-file-path={str(mock_data_directory / 'Answers.csv')}",
        "--answers-file-encoding=ISO8859-1",
        f"--questions-file-path={str(mock_data_directory / 'Questions.csv')}",
        "--questions-file-encoding=ISO8859-1",
        f"--tags-file-path={str(mock_data_directory / 'Tags.csv')}",
        "--tags-file-encoding=ISO8859-1",
        "--save-model",
        f"--save-path={str(tmp_path)}",
        "--test-size=0.01",
    ]
    test_runner.invoke(app, args, catch_exceptions=False)

    assert len(list(tmp_path.glob("*.pkl"))) == 2
