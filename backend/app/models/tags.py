from typing import List

from camel_converter.pydantic_base import CamelBase


class Sentance(CamelBase):
    sentence: str


class TaggedSentence(Sentance):
    tags: List[str]
