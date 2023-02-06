import json
import os
from typing import Any, Iterable, List, Optional, Set, Union

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, FrozenPhrases, Phrases
from pydantic import BaseModel

from datawords import constants, parsers, utils


class PhrasesModelMeta(BaseModel):
    name: str
    lang: str
    parser_conf: parsers.ParserConf
    min_count: int = 1
    threshold: int = 1


class W2VecMeta(BaseModel):
    name: str
    lang: str
    parser_conf: parsers.ParserConf
    phrases_model_path: Optional[str] = None
    epoch: int = 5
    size: int = 100
    window: int = 5
    min_count: int = 1


class PhrasesModel:
    def __init__(
        self,
        parser_conf: parsers.ParserConf,
        min_count=1,
        threshold=1,
        connector_words=None,
        model=None,
    ):

        self._parser_conf = parser_conf
        self._min_count = min_count
        self._threshold = threshold
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
        sentences = parsers.SentencesIterator(X, parser=self._parser)
        _model = Phrases(
            sentences,
            min_count=self._min_count,
            threshold=1,
            connector_words=self._connector_words,
        )
        self._model = _model.freeze()

    def transform(self, X: Iterable):
        results = []
        for txt in X:
            r = self._model[txt.split()]
            results.append(r)
        return results

    def parse(self, txt: str):
        r = self._model[txt.split()]
        return r

    def save(self, fp: Union[str, os.PathLike]):
        name = str(fp).rsplit("/", maxsplit=1)[1]
        utils.mkdir_p(fp)

        conf = PhrasesModelMeta(
            name=name,
            lang=self._parser_conf.lang,
            parser_conf=self._parser_conf,
            min_count=self._min_count,
            threshold=self._threshold,
        )
        self._model.save(f"{fp}/{name}.bin")
        with open(f"{fp}/{name}.json", "w") as f:
            f.write(conf.json())

    @classmethod
    def load(cls, fp: Union[str, os.PathLike]) -> "PhrasesModel":
        name = str(fp).rsplit("/", maxsplit=1)[1]
        with open(f"{fp}/{name}.json", "r") as f:
            jmeta = json.loads(f.read())
            meta = PhrasesModelMeta(**jmeta)
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
        size=100,
        window=5,
        min_count=1,
        workers=1,
        epoch=5,
        model: Word2Vec = None,
        using_kv=False,
    ):

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
        sentences = parsers.SentencesIterator(X, parser=self._parser)
        model = Word2Vec(
            sentences=sentences,
            vector_size=self._size,
            window=self._window,
            min_count=self._min_count,
            workers=self._workers,
        )
        self.model = model

    def parse(self, sentence: str):
        return self._parser.parse(sentence)

    def encode(self, sentence: str):
        """
        gets a sentence in plain text and encode it as vector
        """
        words = self.parse(sentence)
        v = self.vectorize(words)
        return v

    def vectorize(self, sentence: List[str]):
        """Get a vector from a list of words
        if a sentence has words that don't match in the word2vec model,
        then it fills with zeros
        """
        vectors = []
        for word in sentence:
            try:
                _vect = self.model[word]
                vectors.append(_vect)
            except KeyError:
                pass
        if len(vectors) > 0:
            return np.sum(np.array(vectors), axis=0) / (len(vectors) + 0.001)
        return np.zeros((self._size,))

    def save(self, fp: Union[str, os.PathLike]):
        name = str(fp).rsplit("/", maxsplit=1)[1]
        utils.mkdir_p(fp)

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
            f.write(conf.json())

    @classmethod
    def load(cls, fp: Union[str, os.PathLike], keyed_vectors=False) -> "Word2VecHelper":

        name = str(fp).rsplit("/", maxsplit=1)[1]
        if keyed_vectors:
            model = KeyedVectors.load(f"{fp}/{name}.kv")
        else:
            model = Word2Vec.load(f"{fp}/{name}.bin")
        with open(f"{fp}/{name}.json", "r") as f:
            jmeta = json.loads(f.read())
            meta = W2VecMeta(**jmeta)
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
        )
        return obj
