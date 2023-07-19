def open_texts():
    with open("tests/texts.txt", "r") as f:
        texts = f.readlines()
    for t in texts:
        _t = t.strip()
        if _t:
            yield _t


