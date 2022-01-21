import csv

import pytest


@pytest.fixture(scope="session")
def mock_data_directory(tmp_path_factory):
    file_path = tmp_path_factory.mktemp("data")

    with open(file_path / "Answers.csv", "w") as f:
        field_names = ["Id", "OwnerUserId", "CreationDate", "ParentId", "Score", "Body"]
        rows = [
            {
                "Id": 92,
                "OwnerUserId": 61,
                "CreationDate": "2008-08-01T14:45:37Z",
                "ParentId": 90,
                "Score": 13,
                "Body": '<p><a href=""http://svnbook.red-bean.com/"">Version Control with Subversion</a></p>',
            },
            {
                "Id": 93,
                "OwnerUserId": 64,
                "CreationDate": "2008-09-01T14:45:37Z",
                "ParentId": 91,
                "Score": 14,
                "Body": '<p><a href=""http://test.com/"">test</a></p>',
            },
        ]
        writer = csv.DictWriter(f, fieldnames=field_names)

        writer.writeheader()
        writer.writerows(rows)

    with open(file_path / "Questions.csv", "w") as f:
        field_names = ["Id", "OwnerUserId", "CreationDate", "ClosedDate", "Score", "Title", "Body"]
        rows = [
            {
                "Id": 80,
                "OwnerUserId": 26,
                "CreationDate": "2008-08-01T13:57:07Z",
                "ClosedDate": "NA",
                "Score": 26,
                "Title": "SQLStatement.execute() - multiple queries in one statement",
                "Body": "<p>I have written a database generation script</p>",
            },
            {
                "Id": 81,
                "OwnerUserId": 27,
                "CreationDate": "2008-09-01T13:57:07Z",
                "ClosedDate": "2008-09-01T14:45:37Z",
                "Score": 27,
                "Title": "Test Title",
                "Body": "<p>test</p>",
            },
        ]
        writer = csv.DictWriter(f, fieldnames=field_names)

        writer.writeheader()
        writer.writerows(rows)

    with open(file_path / "Tags.csv", "w") as f:
        field_names = ["Id", "Tag"]
        rows = [
            {"Id": 70, "Tag": "flex"},
            {"Id": 90, "Tag": "svn"},
        ]
        writer = csv.DictWriter(f, fieldnames=field_names)

        writer.writeheader()
        writer.writerows(rows)

    yield file_path
