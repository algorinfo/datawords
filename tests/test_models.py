import tempfile

from gensim.models import KeyedVectors, Word2Vec

from datawords import parsers
from datawords.models import Word2VecHelper
from .shared import open_texts



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

    encoded = kv.encode("hello world")

    assert isinstance(wv.model, Word2Vec)
    assert isinstance(wv2.model, Word2Vec)
    assert isinstance(kv.model, KeyedVectors)
    assert encoded.any()


