from typing import List

from gensim.models.phrases import FrozenPhrases

from datawords import constants, parsers, utils
from datawords.models2 import PhrasesModel


def open_texts():
    with open("tests/texts.txt", "r") as f:
        texts = f.readlines()
    for t in texts:
        yield t


def test_models2_phrases():
    texts = open_texts()
    parser_conf = parsers.ParserConf(lang="en")
    model = PhrasesModel(parser_conf)
    model.fit(texts)
    model.save("test", fp="tests")
    model2 = PhrasesModel.load("test", fp="tests")

    assert isinstance(model._model, FrozenPhrases)
    assert isinstance(model2._model, FrozenPhrases)
