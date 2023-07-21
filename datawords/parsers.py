import json
import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, FrozenSet, Iterable, List, Optional, Set, Union

import cattrs
import emoji
import unidecode
from attrs import define, field
from gensim.models.phrases import FrozenPhrases, Phrases

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
    stemmer_class: Optional[str] = None
    phrases_model_path: Optional[str] = None


@define
class PhrasesModelMeta:
    name: str
    lang: str
    parser_conf: ParserConf
    min_count: float = 1.0
    threshold: Optional[float] = None
    max_vocab_size: Optional[int] = None
    version: str = field(default=_utils.get_version())
    path: Optional[str] = None


@define
class StopWords:
    lang: str
    path: str
    words: FrozenSet[str]
    strip_accents: bool = True


def load_stop(lang="en", *, models_path=None, strip_accents=True) -> StopWords:
    """
    Open a list of stop words.

    If `models_path` is ommited then it will look internally in the
    datawords package. Actually, it supports **en**, **pt** and **es**

    """
    if models_path:
        fp = f"{models_path}/stop_{lang}.txt"
    else:
        fp = f"{_utils.pkg_route()}/files/stop_{lang}.txt"

    with open(fp, "r") as f:
        stopwords = f.readlines()

    if strip_accents:
        stop = [unidecode.unidecode(s.lower().strip()) for s in stopwords]
    else:
        stop = [s.lower().strip() for s in stopwords]

    return StopWords(
        lang=lang, path=models_path, words=frozenset(stop), strip_accents=strip_accents
    )


def load_stop2(lang="en", *, models_path=None, strip_accents=True) -> Set[str]:
    """
    Open a list of stop words.

    If `models_path` is ommited then it will look internally
    for the list of words. Actually, it supports **en**, **pt** and **es**

    """
    warnings.warn("load_stop2() will be deprecated", DeprecationWarning)
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


class PhrasesModel:
    def __init__(
        self,
        parser_conf: ParserConf,
        min_count: float = 1.0,
        threshold: Optional[float] = None,
        max_vocab_size: Optional[int] = None,
        connector_words=None,
        model=None,
        saved_path=None,
    ):
        """

        :param parser_conf: configuration of the parser
        :type parser_conf: parsers.ParserConf
        :param min_count: Ignore all words and bigrams with total collected count
            lower than this value.
        :type min_count: float
        :param threshold: Represent a score threshold for forming the phrases
            (higher means fewer phrases). A phrase of words a followed by b is
            accepted if the score of the phrase is greater than threshold. Heavily
            depends on concrete scoring-function, see the scoring parameter.
        :param max_vocab_size: Maximum size (number of tokens) of the vocabulary.
            Used to control pruning of less common words, to keep memory under
            control. The default of 40M needs about 3.6GB of RAM.
            Increase/decrease max_vocab_size depending on how much available
            memory you have.
        :type max_vocab_size: Optional[int] int
        :param connector_words: Set of words that may be included within a phrase,
             without affecting its scoring. If any is provided it will use the
             lang value from the parser_conf. By default datawords
             include CONNECTOR_WORDS for English, Portugues an Spanish.
        :type connector_words: Frozenset[str]

        """

        self._parser_conf = parser_conf
        self._min_count = min_count
        self._threshold = threshold
        self._max_vocab_size = max_vocab_size
        self._connector_words = (
            connector_words or constants.CONNECTOR_WORDS[parser_conf.lang]
        )
        self._model = model
        self._parser = parser_from_conf(parser_conf)
        self._saved_path: Union[str, os.PathLike, None] = saved_path

    @property
    def model(self) -> FrozenPhrases:
        return self._model

    def fit(self, X: Iterable):
        """
        This will train the phrase model. It needs an iterable.

        :param X: An iterable which returns plain texts.
        :type X: Iterable
        """
        sentences = SentencesIterator(X, parser=self._parser)
        _model = Phrases(
            sentences,
            min_count=self._min_count,
            threshold=1,
            connector_words=self._connector_words,
        )
        self._model = _model.freeze()

    def parse(self, txt: str) -> List[str]:
        """
        It will parse only one text.
        :param txt: str
        :return: a list of words
        :rtype: List[str]
        """
        r = self._model[self._parser.parse(txt)]
        return r

    def export_conf(self) -> PhrasesModelMeta:
        name = "phrases_model"
        if self._saved_path:
            name = str(self._saved_path).rsplit("/", maxsplit=1)[1]

        conf = PhrasesModelMeta(
            name=name,
            lang=self._parser_conf.lang,
            parser_conf=self._parser_conf,
            min_count=self._min_count,
            threshold=self._threshold,
            path=self._saved_path,
        )
        return conf

    def save(self, fp: Union[str, os.PathLike]):
        """
        Save phrase model to a folder.

        :param fp: The path to the folder. Each model is stored in a folder.
            The path should be to that folder.
        :type fp: Union[str, os.PathLike]
        """

        self._saved_path = fp
        name = str(fp).rsplit("/", maxsplit=1)[1]
        _utils.mkdir_p(fp)

        conf = self.export_conf()
        self._model.save(f"{fp}/{name}.bin")
        with open(f"{fp}/{name}.json", "w") as f:
            f.write(_utils.asjson(conf))

    @classmethod
    def load(cls, fp: Union[str, os.PathLike]) -> "PhrasesModel":
        """loads the TextIndex model.

        :param fp: The path to the index. Each model is stored in a folder.
             The path should be to that folder.
        :type fp: Union[str, os.PathLike]
        :return: PhrasesModel loaded.
        :rtype: PhrasesModel
        """

        name = str(fp).rsplit("/", maxsplit=1)[1]
        with open(f"{fp}/{name}.json", "r") as f:
            jmeta = json.loads(f.read())
            meta = cattrs.structure(jmeta, PhrasesModelMeta)
        model = Phrases.load(f"{fp}/{name}.bin")
        obj = cls(
            parser_conf=meta.parser_conf,
            min_count=meta.min_count,
            threshold=meta.threshold,
            model=model,
        )
        return obj


def regex_parser(reg_expr, word, stopw: FrozenSet[str]) -> str:
    _found = re.findall(reg_expr, word)
    final = _found
    if stopw:
        final = []
        for word in _found:
            if word not in stopw:
                final.append(word)
    return final


def _apply_regex(reg_expr, word) -> str:
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
    final = _apply_regex(constants.ALPHANUMERIC_REGEX, tkn)
    return final


def doc_parser(
    txt: str,
    stop_words: FrozenSet[str],
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
                _rgx_parser = constants.ALPHANUMERIC_ACCENT_REGEX
            else:
                _rgx_parser = constants.WORDS_ACCENT_REGEX
            word = _apply_regex(_rgx_parser, word)
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
        stop_words=FrozenSet[str],
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
        """
        It gets a txt which could be a phrase, a doc, or anything
        in between, and then parse it using the phraase model or
        the parser.
        """

        if self._phrases:
            parsed = self._phrases.parse(txt)
        else:
            parsed = doc_parser(
                txt,
                self._stopw,
                stemmer=self._stemmer,
                emo_codes=self._emo_codes,
                strip_accents=self._strip_accents,
                lower=self._lower,
                numbers=self._numbers,
                parse_urls=self._parse_urls,
            )
        return parsed

    def export_conf(self) -> ParserConf:
        """
        It exports the parser configuration but omits stopw path and stemmer path.
        """
        phrases_model_path = None
        if self._phrases:
            conf = self._phrases.export_conf()
            phrases_model_path = conf.path
        return ParserConf(
            lang=self._lang,
            emo_codes=self._emo_codes,
            strip_accents=self._strip_accents,
            lower=self._lower,
            numbers=self._numbers,
            parse_urls=self._parse_urls,
            phrases_model_path=phrases_model_path,
        )


def parser_from_conf(
    conf: ParserConf,
    *,
    stopw: Optional[StopWords] = None,
    phrases: Optional[PhrasesModel] = None,
):
    """
    It loads :class:`SentencesParser` based on the :class:`ParserConf` configuration.
    Also it's possible give an already intialized stop words and phrases objects,
    to avoid multiple instances in a same process. 
    """
    _stopw = stopw or load_stop(lang=conf.lang, models_path=conf.stopw_path)
    if not phrases and conf.phrases_model_path:
        phrases = PhrasesModel.load(conf.phrases_model_path)
    return SentencesParser(
        lang=conf.lang,
        phrases_model=phrases,
        lower=conf.lower,
        emo_codes=conf.emo_codes,
        strip_accents=conf.strip_accents,
        numbers=conf.numbers,
        stop_words=_stopw.words,
        parse_urls=conf.parse_urls,
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
