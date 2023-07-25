import json
import os
from typing import Iterable, List, Optional, Union

import cattrs
import numpy as np
from attrs import define, field
from gensim.models import KeyedVectors, Word2Vec

from datawords import _utils, parsers


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
    version: str = field(default=_utils.get_version())
    path: Optional[str] = None


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
        stopw: Optional[parsers.StopWords] = None,
    ):
        """
        It's a wrapper around the original implementation of Word2Vec from the Gensim library.
        It adds the option to store and track the training params of the model
        including the parser used to do so. 

        """

        self._parser_conf = parser_conf
        self._phrases = phrases_model
        # self._models_path = models_path
        self._min_count = min_count
        self._size = size
        self._window = window
        self._min_count = min_count
        self._workers = workers
        self._epoch = epoch
        self._parser = parsers.parser_from_conf(parser_conf, stopw=stopw, phrases=phrases_model)
        self.model: Union[Word2Vec, KeyedVectors] = model
        self._using_kv = using_kv
        self.loaded_from = loaded_from
        self._saved_path = loaded_from

    @property
    def vector_size(self) -> int:
        return self._size

    @property
    def wv(self) -> Union[Word2Vec, KeyedVectors]:
        if self._using_kv:
            return self.model
        else:
            return self.model.wv

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
                _vect = self.wv[word]
                vectors.append(_vect)
            except KeyError:
                pass
        if len(vectors) > 0:
            return np.sum(np.array(vectors), axis=0) / (len(vectors) + 0.001)
        return np.zeros((self._size,))

    def export_conf(self) -> W2VecMeta:
        name="word2vec"
        if self._saved_path:
            name = str(self._saved_path).rsplit("/", maxsplit=1)[1]
        
        conf = W2VecMeta(
            name=name,
            lang=self._parser_conf.lang,
            parser_conf=self._parser_conf,
            size=self._size,
            window=self._window,
            min_count=self._min_count,
            epoch=self._epoch,
            path=self._saved_path,
        )
        return conf
        

    def save(self, fp: Union[str, os.PathLike]):
        name = str(fp).rsplit("/", maxsplit=1)[1]
        _utils.mkdir_p(fp)
        self._saved_path = fp
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
            phrases = parsers.PhrasesModel.load(meta.phrases_model_path)
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
