import json
import os
from typing import Iterable, List, Optional, Union

import cattrs
import numpy as np
from attrs import define
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.phrases import FrozenPhrases, Phrases

from datawords import _utils, constants, parsers


@define
class PhrasesModelMeta:
    name: str
    lang: str
    parser_conf: parsers.ParserConf
    min_count: float = 1.0
    threshold: Optional[float] = None
    max_vocab_size: Optional[int] = None
    version: str = _utils.get_version()


@define
class W2VecMeta:
    name: str
    lang: str
    parser_conf: parsers.ParserConf
    phrases_model_path: Optional[str] = None
    epoch: int = 5
    size: int = 100
    window: int = 5
    min_count: int = 1
    version: str = _utils.get_version()


class PhrasesModel:
    def __init__(
        self,
        parser_conf: parsers.ParserConf,
        min_count: float = 1.0,
        threshold: Optional[float] = None,
        max_vocab_size: Optional[int] = None,
        connector_words=None,
        model=None,
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
        self._parser = self._parser_factory()

    @property
    def model(self) -> FrozenPhrases:
        return self._model

    def _load_stopw(self):
        return parsers.load_stop2(
            self._parser_conf.lang, models_path=self._parser_conf.stopw_path
        )

    def _parser_factory(self) -> parsers.SentencesParser:
        stopw = self._load_stopw()
        return parsers.SentencesParser(
            lang=self._parser_conf.lang,
            lower=self._parser_conf.lower,
            emo_codes=self._parser_conf.emo_codes,
            strip_accents=self._parser_conf.strip_accents,
            numbers=self._parser_conf.numbers,
            stop_words=stopw,
        )

    def fit(self, X: Iterable):
        """
        This will train the phrase model. It needs an iterable.

        :param X: An iterable which returns plain texts.
        :type X: Iterable
        """
        sentences = parsers.SentencesIterator(X, parser=self._parser)
        _model = Phrases(
            sentences,
            min_count=self._min_count,
            threshold=1,
            connector_words=self._connector_words,
        )
        self._model = _model.freeze()

    def transform(self, X: Iterable) -> List[List[str]]:
        """
        Trasform a list of texts.

        :param X: an iterable.
        :type X: Iterable

        :return: A list of phrases.
        :rtype: List[List[str]]
        """
        results = []
        for txt in X:
            r = self._model[txt.split()]
            results.append(r)
        return results

    def parse(self, txt: str) -> List[str]:
        """
        It will parse only one text.
        :param txt: str
        :return: a list of words
        :rtype: List[str]
        """
        r = self._model[txt.split()]
        return r

    def save(self, fp: Union[str, os.PathLike]):
        """
        Save phrase model to a folder.

        :param fp: The path to the folder. Each model is stored in a folder.
            The path should be to that folder.
        :type fp: Union[str, os.PathLike]
        """

        name = str(fp).rsplit("/", maxsplit=1)[1]
        _utils.mkdir_p(fp)

        conf = PhrasesModelMeta(
            name=name,
            lang=self._parser_conf.lang,
            parser_conf=self._parser_conf,
            min_count=self._min_count,
            threshold=self._threshold,
        )
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


class Word2VecHelper:
    def __init__(
        self,
        parser_conf: parsers.ParserConf,
        phrases_model=None,
        size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 1,
        epoch: int = 5,
        model: Word2Vec = None,
        using_kv=False,
        loaded_from=None,
    ):
        """

        :param parser_conf: configuration of the parser
        :type parser_conf: parsers.ParserConf
        :param min_count: Ignore all words and bigrams with total collected
            count lower than this value.
        :type min_count: Optional[float]
        :param threshold: Represent a score threshold for forming the phrases
            (higher means fewer phrases). A phrase of words a followed by b is
            accepted if the score of the phrase is greater than threshold.
            Heavily depends on concrete scoring-function, see the scoring
            parameter.
        :param max_vocab_size: Maximum size (number of tokens) of the
            vocabulary. Used to control pruning of less common words,
            to keep memory under control. The default of 40M needs
            about 3.6GB of RAM. Increase/decrease max_vocab_size
            depending on how much available memory you have.
        :type max_vocab_size: Optional[int] int
        :param connector_words: Set of words that may be included within
           a phrase, without affecting its scoring. If any is provided it
           will use the lang value from the parser_conf. By default
           datawords include CONNECTOR_WORDS for English, Portugues
           an Spanish.
        :type connector_words: Frozenset[str]

        """

        self._parser_conf = parser_conf
        self._stopw = self._load_stopw(parser_conf.lang, self._parser_conf.stopw_path)
        self._phrases = phrases_model
        # self._models_path = models_path
        self._min_count = min_count
        self._size = size
        self._window = window
        self._min_count = min_count
        self._workers = workers
        self._epoch = epoch
        self._parser = self._parser_factory()
        self.model: Union[Word2Vec, KeyedVectors] = model
        self._using_kv = using_kv
        self.loaded_from = loaded_from

    @property
    def vector_size(self) -> int:
        return self._size

    @property
    def wv(self) -> Union[Word2Vec, KeyedVectors]:
        return self.model

    def _parser_factory(self) -> parsers.SentencesParser:
        return parsers.SentencesParser(
            lang=self._parser_conf.lang,
            lower=self._parser_conf.lower,
            phrases_model=self._phrases,
            emo_codes=self._parser_conf.emo_codes,
            strip_accents=self._parser_conf.strip_accents,
            numbers=self._parser_conf.numbers,
            stop_words=self._stopw,
        )

    def _load_stopw(self, lang, models_path):
        return parsers.load_stop2(lang, models_path=models_path)

    def fit(self, X: Iterable):
        """
        This will train the model. It needs an iterable.

        :param X: An iterable which returns plain texts.
        :type X: Iterable
        """

        sentences = parsers.SentencesIterator(X, parser=self._parser)
        model = Word2Vec(
            sentences=sentences,
            vector_size=self._size,
            window=self._window,
            min_count=self._min_count,
            workers=self._workers,
        )
        self.model = model

    # def transform(self, X: Iterable) -> np.ndarray:
    #     """
    #     This will train the model. It needs an iterable.

    #     :param X: An iterable which returns plain texts.
    #     :type X: Iterable
    #     """

    #     sentences = parsers.SentencesIterator(X, parser=self._parser)
    #     vectors = [self.encode(st) for st in sentences]
    #     return np.asarray(vectors)

    # def fit_transform(self, X: Iterable) -> np.ndarray:
    #     self.fit(X)
    #     return self.transform(X)

    def parse(self, sentence: str) -> List[str]:
        """
        It will parse only one text.
        :param txt: str
        :return: a list of words
        :rtype: List[str]
        """
        return self._parser.parse(sentence)

    def encode(self, sentence: str) -> np.ndarray:
        """
        gets a sentence in plain text and encode it as vector
        """
        words = self.parse(sentence)
        v = self.vectorize(words)
        return v

    def vectorize(self, sentence: List[str]) -> np.ndarray:
        """Get a vector from a list of words
        if a sentence has words that don't match in the word2vec model,
        then it fills with zeros
        """
        vectors = []
        for word in sentence:
            try:
                _vect = self.model.wv[word]
                vectors.append(_vect)
            except KeyError:
                pass
        if len(vectors) > 0:
            return np.sum(np.array(vectors), axis=0) / (len(vectors) + 0.001)
        return np.zeros((self._size,))

    def save(self, fp: Union[str, os.PathLike]):
        name = str(fp).rsplit("/", maxsplit=1)[1]
        _utils.mkdir_p(fp)
        conf = W2VecMeta(
            name=name,
            lang=self._parser_conf.lang,
            parser_conf=self._parser_conf,
            size=self._size,
            window=self._window,
            min_count=self._min_count,
            epoch=self._epoch,
        )

        self.model.save(f"{fp}/{name}.bin")
        self.model.wv.save(f"{fp}/{name}.kv")
        with open(f"{fp}/{name}.json", "w") as f:
            f.write(_utils.asjson(conf))

    @classmethod
    def load(cls, fp: Union[str, os.PathLike], keyed_vectors=False) -> "Word2VecHelper":
        name = str(fp).rsplit("/", maxsplit=1)[1]
        if keyed_vectors:
            model = KeyedVectors.load(f"{fp}/{name}.kv", mmap="r")
        else:
            model = Word2Vec.load(f"{fp}/{name}.bin")
        with open(f"{fp}/{name}.json", "r") as f:
            jmeta = json.loads(f.read())
            meta = cattrs.structure(jmeta, W2VecMeta)
        phrases = None
        if meta.phrases_model_path:
            phrases = PhrasesModel.load(meta.phrases_model_path)
        obj = cls(
            parser_conf=meta.parser_conf,
            phrases_model=phrases,
            size=meta.size,
            window=meta.window,
            min_count=meta.min_count,
            model=model,
            using_kv=keyed_vectors,
            loaded_from=str(fp),
        )
        return obj
