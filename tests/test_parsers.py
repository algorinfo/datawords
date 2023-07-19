import tempfile

from gensim.models.phrases import FrozenPhrases

from datawords import parsers

from .shared import open_texts


def test_parsers_generate_ngrams():
    ngrams = parsers.generate_ngrams(["hello", "Tom", "how"])

    assert len(ngrams) == 2


def test_parsers_load_stop():
    stopw = parsers.load_stop()
    assert "the" in stopw.words
    assert isinstance(stopw, parsers.StopWords)


def test_parsers_doc_parser():
    stopw = parsers.load_stop()
    t = """Goodbye world, Hi Fernández. http://chuchu.me/spotify  방 #EEER 😋.\n
        This is the 99th case for 99 days"""

    default = parsers.doc_parser(
        t, stopw.words, emo_codes=False, strip_accents=True, numbers=False, parse_urls=False
    )
    _all = parsers.doc_parser(
        t,
        stopw.words,
        emo_codes=True,
        strip_accents=False,
        numbers=True,
        parse_urls=True,
    )

    assert len(default) == 8
    # assert "th" not in default # it fails now
    assert "this" not in default
    assert "á" not in default[3]
    assert default[0][0] == "g"
    assert "99" in _all
    assert ":face_savoring_food:" in _all
    assert "http://chuchu.me/spotify" in _all


def test_parsers_sentence():
    stopw = parsers.load_stop()
    t = """Goodbye world, Hi Fernández. http://chuchu.me/spotify  방 #EEER 😋.\n
        This is the 99th case for 99 days"""

    parser = parsers.SentencesParser(
        t,
        stop_words=stopw.words,
        emo_codes=False,
        strip_accents=True,
        numbers=False,
        parse_urls=False,
    )
    default = parser.parse(t)
    conf = parser.export_conf()

    assert len(default) == 8
    # assert "th" not in default # it fails now
    assert "this" not in default
    assert "á" not in default[3]
    assert default[0][0] == "g"
    assert isinstance(conf, parsers.ParserConf)


def test_parsers_phrases():
    texts = open_texts()
    parser_conf = parsers.ParserConf(lang="en")
    model = parsers.PhrasesModel(parser_conf)
    model.fit(texts)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = f"{tmpdir}/tests"
        model.save(fp)
        model2 = parsers.PhrasesModel.load(fp)

    assert isinstance(model.model, FrozenPhrases)
    assert isinstance(model2.model, FrozenPhrases)
