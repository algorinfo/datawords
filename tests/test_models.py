import tempfile

from gensim.models import KeyedVectors, Word2Vec
from gensim.models.phrases import FrozenPhrases

from datawords import parsers
from datawords.models import PhrasesModel, Word2VecHelper


def open_texts():
    with open("tests/texts.txt", "r") as f:
        texts = f.readlines()
    for t in texts:
        _t = t.strip()
        if _t:
            yield _t



def test_models_phrases():
    texts = open_texts()
    parser_conf = parsers.ParserConf(lang="en")
    model = PhrasesModel(parser_conf)
    model.fit(texts)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = f"{tmpdir}/tests"
        model.save(fp)
        model2 = PhrasesModel.load(fp)

    assert isinstance(model.model, FrozenPhrases)
    assert isinstance(model2.model, FrozenPhrases)


def test_models_word2vec():
    texts = open_texts()
    parser_conf = parsers.ParserConf(lang="en")
    # model = PhrasesModel(parser_conf)
    # model.fit(texts)
    wv = Word2VecHelper(parser_conf)
    wv.fit(texts)
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = f"{tmpdir}/tests"
        wv.save(fp)
        wv2 = Word2VecHelper.load(fp)
        kv = Word2VecHelper.load(fp, keyed_vectors=True)

    assert isinstance(wv.model, Word2Vec)
    assert isinstance(wv2.model, Word2Vec)
    assert isinstance(kv.model, KeyedVectors)
