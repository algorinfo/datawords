import re
import warnings
from typing import Any, FrozenSet, Iterable, Optional, Set, List
from abc import ABC, abstractmethod

import emoji
import unidecode
from pydantic import BaseModel

from datawords import _utils


class ParserProto(ABC):
    @abstractmethod
    def parse(self, txt: str) -> List[str]:
        pass


class ParserConf(BaseModel):
    lang: str = "en"
    emo_codes: bool = False
    strip_accents: bool = False
    lower: bool = True
    numbers: bool = True
    stopw_path: Optional[str] = None
    use_stemmer: bool = False
    phrases_model_path: Optional[str] = None


WORDS_REGEX = r"[a-zA-Z]+"


def load_stop(base_path, lang="en", strip_accents=True) -> Set[str]:
    """
    Open a list of stop words.
    The final path will be:
         models/stop_es.txt
    """
    warnings.warn("load_stop() will be deprecated", DeprecationWarning)
    with open(f"{base_path}/stop_{lang}.txt", "r") as f:
        stopwords = f.readlines()

    if strip_accents:
        stop = {unidecode.unidecode(s.strip()) for s in stopwords}
    else:
        stop = {s.strip() for s in stopwords}

    # logger.debug("%d stop words loaded for lang %s", len(stop), lang)

    return stop


def load_stop2(lang="en", *, models_path=None, strip_accents=True) -> Set[str]:
    """
    Open a list of stop words.
    The final path will be:
         models/stop_es.txt
    """
    if models_path:
        fp = f"{models_path}/stop_{lang}.txt"
    else:
        fp = f"{_utils.pkg_route()}/files/stop_{lang}.txt"

    with open(fp, "r") as f:
        stopwords = f.readlines()

    if strip_accents:
        stop = {unidecode.unidecode(s.strip()) for s in stopwords}
    else:
        stop = {s.strip() for s in stopwords}

    # logger.debug("%d stop words loaded for lang %s", len(stop), lang)

    return stop


def doc_parser(
    txt: str,
    stop_words: Set[str],
    stemmer: Optional[Any] = None,
    emo_codes=False,
    strip_accents=False,
    lower=True,
    numbers=True,
) -> List[str]:
    """
    Get a string text an return a list of words
    # from nltk.stem import SnowballStemmer
    """
    text = re.sub(r"<br>+", "", txt)
    _doc = []
    for tkn in text.split():
        # ("<[^>]*>", "")
        word = None
        _matched = re.search(r"\w+", tkn)
        if _matched:
            word = _matched.group()

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


class SentencesParser(ParserProto):
    def __init__(
        self,
        lang="en",
        lower=True,
        phrases_model=None,
        stemmer: Optional[Any] = None,
        emo_codes=False,
        strip_accents=False,
        numbers=True,
        stop_words=Set[str],
    ):
        self._lang = lang
        self._lower = lower
        self._phrases = phrases_model
        self._stemmer = stemmer
        self._emo_codes = emo_codes
        self._strip_accents = strip_accents
        self._numbers = numbers
        self._stopw = stop_words

    def parse(self, txt: str) -> List[str]:
        words = doc_parser(
            txt,
            self._stopw,
            stemmer=self._stemmer,
            emo_codes=self._emo_codes,
            strip_accents=self._strip_accents,
            lower=self._lower,
            numbers=self._numbers,
        )
        if self._phrases:
            return self._phrases[words]
        return words


class SentencesIterator:
    def __init__(self, data: Iterable, *, parser: ParserProto):
        self._parser = parser
        self._data = data

    def parser_generator(self):
        for txt in self._data:
            yield self._parser.parse(txt)

    def __iter__(self):
        self._generator = self.parser_generator()
        return self

    def __next__(self):
        result = next(self._generator)
        if not result:
            raise StopIteration
        return result
