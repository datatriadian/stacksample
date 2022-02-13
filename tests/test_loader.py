from stacksample.loader import load_answers, load_questions, load_tags


def test_load_answers(mock_data_directory):
    df = load_answers(mock_data_directory / "Answers.csv", encoding="ISO8859-1")
    assert df.iloc[0]["Id"] == 92


def test_load_questions(mock_data_directory):
    df = load_questions(mock_data_directory / "Questions.csv", encoding="ISO8859-1")
    assert df.iloc[0]["Id"] == 80


def test_load_tags(mock_data_directory):
    df = load_tags(mock_data_directory / "Tags.csv", encoding="ISO8859-1")
    assert df.iloc[0]["Id"] == 80
