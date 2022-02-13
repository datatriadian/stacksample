import pandas as pd

from stacksample import training


def test_composit_labels():
    data = {"id": [1, 1, 2, 2], "tag": ["javascript", "html", "html", "javascript"]}
    df = pd.DataFrame(data=data)
    tags = training._create_composit_labels(df)
    grouped_tags = tags.groupby("tag").size().reset_index(name="count")

    assert grouped_tags["tag"][0] == "html, javascript"
    assert grouped_tags["count"][0] == 2
