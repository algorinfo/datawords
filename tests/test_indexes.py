from pathlib import Path
from annoy import AnnoyIndex
from tempfile import mkdtemp
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


def test_indexes_lite_doc():
    l1 = LiteDoc(id="test", text="pepe")
    l2 = LiteDoc(id="test", text="other text")
    l3 = LiteDoc(id="different", text="other text")
    s = set()
    s.add(l1)
    s.add(l2)
    s.add(l3)
    assert len(s) == 2
    assert l1 == l2
    assert l1 != l3

def test_indexes_sqlite_get():
    l1 = LiteDoc(id="pepe", text="pepe")
    ix = SQLiteIndex()
    ix.add_batch([l1])
    doc = ix.get_doc("pepe")

    assert doc.text == "pepe"

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
    tmp = mkdtemp()
    ix.save(tmp)
    ix2 = TextIndex.load(tmp, words_model=wv)
    
    assert isinstance(ix.ix, AnnoyIndex)
    assert isinstance(ix2, TextIndex)


def test_indexes_sqlite_search():
    texts = list(open_texts())
    stopw = parsers.load_stop2()
    docs = [LiteDoc(id=ix, text=t) for ix, t in enumerate(texts[:5])]
    ix = SQLiteIndex(stopwords=stopw)
    ix.add_batch(docs)
    total = ix.total
    res = ix.search("coco", top_n=1)
    assert isinstance(res[0], LiteDoc)
    assert total == 5

def test_indexes_sqlite_build():
    elements = {x[0]: x[1] for x in enumerate(list(open_texts())[:5])}

    def getter(id_):
        return elements[id_]
    stopw = parsers.load_stop2()
    # docs = [LiteDoc(id=ix, text=t) for ix, t in enumerate(texts[:5])]
    ids = list(elements.keys())
    ix = SQLiteIndex.build(ids=ids, getter=getter, stopwords=stopw)
    total = ix.total
    res = ix.search("coco", top_n=1)
    assert isinstance(res[0], LiteDoc)
    assert total == 5


def test_indexes_sqlite_search_query():
    elements = {x[0]: x[1] for x in enumerate(list(open_texts())[:5])}

    def getter(id_):
        return elements[id_]
    stopw = parsers.load_stop2()
    # docs = [LiteDoc(id=ix, text=t) for ix, t in enumerate(texts[:5])]
    ids = list(elements.keys())
    ix = SQLiteIndex.build(ids=ids, getter=getter, stopwords=stopw)
    res = ix.search_query("cocomelon AND goal", top_n=1)
    assert isinstance(res[0], LiteDoc)
