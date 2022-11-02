import re
from typing import Any, Optional, Set

import emoji
import unidecode

SPANISH_CONNECTOR_WORDS = ["un", "una", "y", "para", "por", "en", "el", "asi", "con", "si", "asi", "aunque", "de",
                           "sino", "pero", "sin", "como", "segun", "ahora", "que", "cuando", "durante", "entonces", "hasta", "luego"]

WORDS_REGEX = r"[a-zA-Z]+"


def load_stop(base_path,
              lang="en", strip_accents=True) -> Set[str]:
    """
    Open a list of stop words.
    The final path will be:
         models/stop_es.txt
    """
    with open(f'{base_path}/stop_{lang}.txt', 'r') as f:
        stopwords = f.readlines()

    if strip_accents:
        stop = {unidecode.unidecode(s.strip())
                for s in stopwords}
    else:
        stop = {s.strip() for s in stopwords}

    # logger.debug("%d stop words loaded for lang %s", len(stop), lang)

    return stop


def doc_parser(txt: str, stop_words: Set[str],
               stemmer: Optional[Any] = None,
               emo_codes=False,
               strip_accents=False,
               lower=True,
               numbers=True
               ):
    """
    Get a string text an return a list of words
    # from nltk.stem import SnowballStemmer
    """
    text = re.sub(r"<br>+", "", txt)
    _doc = []
    for tkn in text.split():
        try:
            ("<[^>]*>", "")
            word = re.search(r"\w+", tkn).group()
        except AttributeError:
            word = None

        if word:
            word_norm = unidecode.unidecode(word.strip()).lower()
            if not numbers:
                _letters = re.findall(WORDS_REGEX, word)
                word = _letters[0]
            if lower:
                word = word.lower()
            final_word = word
            if strip_accents:
                final_word = unidecode.unidecode(word.strip())

            if word_norm not in stop_words:
                if stemmer:
                    final_word = stemmer.stem(final_word)
                _doc.append(final_word)

    if emo_codes:
        _codes = emoji.demojize(text)
        _emojis = re.findall(r"\:[a-zA-Z_-]+\:", _codes)
        _doc.extend(_emojis)

    return _doc
