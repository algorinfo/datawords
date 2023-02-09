from annoy import AnnoyIndex

from datawords import parsers
from datawords.indexes import LiteDoc, SQLiteIndex, TextIndex
from datawords.models import Word2VecHelper


def open_texts():
    with open("tests/texts.txt", "r") as f:
        texts = f.readlines()
    for t in texts:
        _t = t.strip()
        if _t:
            yield _t


def test_indexes_words_index():
    texts = open_texts()
    parser_conf = parsers.ParserConf(lang="en")
    # model = PhrasesModel(parser_conf)
    # model.fit(texts)
    elements = {x[0]: x[1] for x in enumerate(list(open_texts()))}

    def getter(id_):
        return elements[id_]

    wv = Word2VecHelper(parser_conf)
    wv.fit(texts)
    # WordsIndex(words_model=wv, id_mapper={})
    ids = list(elements.keys())
    ix = TextIndex.build(ids, getter=getter, words_model=wv)
    assert isinstance(ix.ix, AnnoyIndex)


def test_indexes_sqlite_search():
    texts = list(open_texts())
    stopw = parsers.load_stop2()
    docs = [LiteDoc(id=ix, text=t) for ix, t in enumerate(texts[:5])]
    ix = SQLiteIndex(stopwords=stopw)
    ix.add_batch(docs)
    total = ix.total
    res = ix.search("coco", limit=1)
    assert isinstance(res[0], LiteDoc)
    assert total == 5
