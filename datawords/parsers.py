import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Set

import emoji
import unidecode
from attrs import define

from datawords import _utils, constants


class ParserProto(ABC):
    """Abstract class that any parser should agree with."""

    @abstractmethod
    def parse(self, txt: str) -> List[str]:
        pass


@define
class ParserConf:
    """
    Related to :meth:`doc_parser`
    """

    lang: str = "en"
    emo_codes: bool = False
    strip_accents: bool = False
    lower: bool = True
    numbers: bool = True
    parse_urls: bool = False
    stopw_path: Optional[str] = None
    use_stemmer: bool = False
    phrases_model_path: Optional[str] = None


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

    If `models_path` is ommited then it will look internally
    for the list of words. Actually, it supports **en**, **pt** and **es**

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


def apply_regex(reg_expr, word) -> str:
    _found = re.findall(reg_expr, word)
    if len(_found) > 0:
        if len(_found) > 1:
            word = "".join(_found)
        else:
            word = _found[0]
        return word
    return ""


def norm_token(tkn: str) -> str:
    """
    An opinated token normalizer. It lower any string, strip any accents
    and keeps letters and number from a token.
    """
    final = unidecode.unidecode(tkn.lower().strip())
    # final = apply_regex(r"[\w]", tkn)
    final = apply_regex(constants.ALPHANUMERIC_REGEX, tkn)
    return final


def doc_parser(
    txt: str,
    stop_words: Set[str],
    stemmer: Optional[Any] = None,
    emo_codes=False,
    strip_accents=True,
    lower=True,
    numbers=False,
    parse_urls=False,
) -> List[str]:
    """
    Get a string text an return a list of words.

    .. note::
        emo_codes and parse_urls alter the order of the tokens.
        If they are found, then it will put them at the end of the list.


    This function is related to :class:`ParserConf`

    :param txt: text to be parsed.
    :type txt: str
    :param stop_words: a list of stop words.
        It's possible to get a list using :meth:`load_stop2`
    :type stop_words: Set[str]
    :param stemmer: optional, a stemmer.
    :type stemmer: Optional[Any]
    :param emo_codes: if true, emo_codes will be decoded into text
    :type emo_codes: bool
    :param stip_accents: replace accents with the same letter without accent
    :type strip_accents: bool
    :param lower: transform text to lower letters.
    :type lower: bool
    :param numbers: if True it will keep numbers.
    :type numbers: bool
    :param parse_urls: keep urls.
    :type parse_urls: bool
    :return: a list of tokens
    :rtype: List[str]

    """
    # from nltk.stem import SnowballStemmer
    # text = re.sub(r"<br>+", "", txt)
    # text = txt
    text = re.sub(constants.URL_REGEX, "", txt, flags=re.MULTILINE)

    if strip_accents:
        text = unidecode.unidecode(text)
    if lower:
        text = text.lower()

    _doc = []

    for tkn in text.split():
        # ("<[^>]*>", "")
        word = tkn
        norm = norm_token(tkn)
        if norm and norm not in stop_words:
            if numbers:
                regex_parser = constants.ALPHANUMERIC_ACCENT_REGEX
            else:
                regex_parser = constants.WORDS_ACCENT_REGEX
            word = apply_regex(regex_parser, word)
            if word:
                if stemmer:
                    word = stemmer.stem(word)
                _doc.append(word)

    if emo_codes:
        _codes = emoji.demojize(text)
        _emojis = re.findall(r"\:[a-zA-Z_-]+\:", _codes)
        _doc.extend(_emojis)

    if parse_urls:
        urls = re.findall(constants.URL_REGEX, txt)
        _doc.extend(urls)

    return _doc


# def generate_ngrams(s: str, n: int = 2, numbers=True, lower=True, sep=" ") -> List[str]:
def generate_ngrams(tokens: List[str], n: int = 2, sep=" ") -> List[str]:
    """
    Generate n grams from a string `s`.

    :param tokens: a list of words already parsed.
    :type tokens: List[str]
    :param n: how many ngrams generate. 2 by default.
    :type n: int
    :param sep: Field to use as seperator between words.
    :type sep: str
    :return: a list of ngrams
    :rtype: List[str]

    """
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [f"{sep}".join(ngram) for ngram in ngrams]


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
        parse_urls=False,
    ):
        self._lang = lang
        self._lower = lower
        self._phrases = phrases_model
        self._stemmer = stemmer
        self._emo_codes = emo_codes
        self._strip_accents = strip_accents
        self._numbers = numbers
        self._stopw = stop_words
        self._parse_urls = parse_urls

    def parse(self, txt: str) -> List[str]:
        words = doc_parser(
            txt,
            self._stopw,
            stemmer=self._stemmer,
            emo_codes=self._emo_codes,
            strip_accents=self._strip_accents,
            lower=self._lower,
            numbers=self._numbers,
            parse_urls=self._parse_urls
        )
        if self._phrases:
            return self._phrases[words]
        return words

    def export_conf(self) -> ParserConf:
        """
        It exports the parser configuration but omits stopw path and stemmer path. 
        """
        return ParserConf(
            lang=self._lang,
            emo_codes=self._emo_codes,
            strip_accents=self._strip_accents,
            lower=self._lower,
            numbers=self._numbers,
            parse_urls=self._parse_urls,
        )


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
