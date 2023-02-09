from datawords import parsers


def open_texts():
    with open("tests/texts.txt", "r") as f:
        texts = f.readlines()
    for t in texts:
        _t = t.strip()
        if _t:
            yield _t


def test_parsers_generate_ngrams():
    ngrams = parsers.generate_ngrams(["hello", "Tom", "how"])

    assert len(ngrams) == 2


def test_parsers_doc_parser():
    stopw = parsers.load_stop2()
    t = """Goodbye world, Hi Fernández. http://chuchu.me/spotify  방 #EEER 😋.\n
        This is the 99th case for 99 days"""

    default = parsers.doc_parser(
        t, stopw, emo_codes=False, strip_accents=True, numbers=False, parse_urls=False
    )
    _all = parsers.doc_parser(
        t, stopw, emo_codes=True, strip_accents=False, numbers=True, parse_urls=True
    )

    assert len(default) == 8
    # assert "th" not in default # it fails now
    assert "this" not in default
    assert "á" not in default[3]
    assert default[0][0] == "g"
    assert "99" in _all
    assert ":face_savoring_food:" in _all
    assert "http://chuchu.me/spotify" in _all
