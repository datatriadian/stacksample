import csv

import pytest
from typer.testing import CliRunner


@pytest.fixture
def test_runner():
    return CliRunner()


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
            {
                "Id": 94,
                "OwnerUserId": 61,
                "CreationDate": "2008-08-01T14:45:37Z",
                "ParentId": 92,
                "Score": 13,
                "Body": '<p><a href=""http://svnbook.red-bean.com/"">Version Control with Subversion</a></p>',
            },
            {
                "Id": 95,
                "OwnerUserId": 64,
                "CreationDate": "2008-09-01T14:45:37Z",
                "ParentId": 93,
                "Score": 14,
                "Body": '<p><a href=""http://test.com/"">test</a></p>',
            },
            {
                "Id": 96,
                "OwnerUserId": 61,
                "CreationDate": "2008-08-01T14:45:37Z",
                "ParentId": 90,
                "Score": 13,
                "Body": '<p><a href=""http://svnbook.red-bean.com/"">Version Control with Subversion</a></p>',
            },
            {
                "Id": 97,
                "OwnerUserId": 64,
                "CreationDate": "2008-09-01T14:45:37Z",
                "ParentId": 91,
                "Score": 14,
                "Body": '<p><a href=""http://test.com/"">test</a></p>',
            },
            {
                "Id": 98,
                "OwnerUserId": 61,
                "CreationDate": "2008-08-01T14:45:37Z",
                "ParentId": 92,
                "Score": 13,
                "Body": '<p><a href=""http://svnbook.red-bean.com/"">Version Control with Subversion</a></p>',
            },
            {
                "Id": 99,
                "OwnerUserId": 64,
                "CreationDate": "2008-09-01T14:45:37Z",
                "ParentId": 93,
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
            {
                "Id": 82,
                "OwnerUserId": 26,
                "CreationDate": "2008-08-01T13:57:07Z",
                "ClosedDate": "NA",
                "Score": 26,
                "Title": "SQLStatement.execute() - multiple queries in one statement",
                "Body": "<p>I have written a database generation script</p>",
            },
            {
                "Id": 83,
                "OwnerUserId": 27,
                "CreationDate": "2008-09-01T13:57:07Z",
                "ClosedDate": "2008-09-01T14:45:37Z",
                "Score": 27,
                "Title": "Test Title",
                "Body": "<p>test</p>",
            },
            {
                "Id": 84,
                "OwnerUserId": 26,
                "CreationDate": "2008-08-01T13:57:07Z",
                "ClosedDate": "NA",
                "Score": 26,
                "Title": "SQLStatement.execute() - multiple queries in one statement",
                "Body": "<p>I have written a database generation script</p>",
            },
            {
                "Id": 85,
                "OwnerUserId": 27,
                "CreationDate": "2008-09-01T13:57:07Z",
                "ClosedDate": "2008-09-01T14:45:37Z",
                "Score": 27,
                "Title": "Test Title",
                "Body": "<p>test</p>",
            },
            {
                "Id": 86,
                "OwnerUserId": 26,
                "CreationDate": "2008-08-01T13:57:07Z",
                "ClosedDate": "NA",
                "Score": 26,
                "Title": "SQLStatement.execute() - multiple queries in one statement",
                "Body": "<p>I have written a database generation script</p>",
            },
            {
                "Id": 87,
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
            {"Id": 80, "Tag": "flex"},
            {"Id": 81, "Tag": "svn"},
            {"Id": 82, "Tag": "flex"},
            {"Id": 83, "Tag": "svn"},
            {"Id": 84, "Tag": "flex"},
            {"Id": 85, "Tag": "svn"},
            {"Id": 86, "Tag": "flex"},
            {"Id": 87, "Tag": "svn"},
        ]
        writer = csv.DictWriter(f, fieldnames=field_names)

        writer.writeheader()
        writer.writerows(rows)

    yield file_path
