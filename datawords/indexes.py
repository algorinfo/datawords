import json
import os
from typing import Any, Callable, Dict, List, Union

import numpy as np
from annoy import AnnoyIndex
from pydantic import BaseModel

from datawords import utils
from datawords.models import Word2VecHelper


class TextIndexMeta(BaseModel):
    name: str
    words_model_path: str
    vector_size: int = 100
    ann_trees: int = 10
    distance_metric: str = "angular"
    version: str = utils.get_version()


class TextIndex:
    def __init__(
        self,
        words_model: Word2VecHelper,
        id_mapper: Dict[int, Any],
        ann_trees: int = 10,
        distance_metric: str = "angular",
        ix: AnnoyIndex = None,
    ):
        """
        It's a wrapper around `Spotify's AnnoyIndex https://github.com/spotify/annoy`_
        This Index is opinated, usually AnnoyIndex allows any kind of Vector,
        insted, this index is prepared to accept only texts, parse it and
        perform searchs.

        AnnoyIndex only accepts integers as indices of the data,
        for that reason and id_mapper is built in the build process.


        :param words_model: Words Model to be used for vectorizing texts.
        :type words_model: Word2VecHelper
        :param id_mapper: A dict where keys are the indices stored in Annoy
        and values are the real indices in the domain which the data belongs to.
        :type id_mapper: Dict[int, Any]
        :param distance_metric: type of distance to be used: "angular",
        "euclidean", "manhattan", "hamming", or "dot".
        :type distance_metric: str
        :param ix: AnnoyIndex object.
        :type ix: AnnoyIndex


        """
        self.ix: AnnoyIndex = ix
        self.id_mapper = id_mapper
        self.words = words_model
        self._ann_trees = ann_trees
        self._distance_metric = distance_metric
        self._vector_size = words_model.vector_size

    def encode(self, txt: str) -> np.ndarray:
        """a wrapper around the encode method from Word2Vec model.
        it's get a txt and return a the vectorized version of the text.

        :param txt: full string sentence.
        :type txt: str
        :return: an array from text
        :rtype: np.ndarray
        """
        return self.words.encode(txt)

    def search(self, txt: str, top_n=5, include_distances=False):
        """
        After AnnoyIndex was trained, it can performs searchs over the index.
        It will returns the original id's using the id_mapper built during the
        training process.

        :param txt: full text to search
        :type txt: str
        :param top_n: How many results it returns.
        :type top_n: int
        :param include_distances: include distances.
        """
        v = self.encode(txt)
        vectors = self.ix.get_nns_by_vector(
            v, top_n, include_distances=include_distances
        )
        if include_distances:
            _res = [self.id_mapper[_v] for _v in vectors[0]]
            res = list(zip(_res, vectors[1]))
        else:
            res = [self.id_mapper[_v] for _v in vectors]
        return res

    @classmethod
    def build(
        cls,
        ids: List[Any],
        *,
        getter: Callable,
        words_model_path: str = None,
        words_model: Word2VecHelper = None,
        distance_metric: str = "angular",
        ann_trees: int = 10,
        n_jobs: int = -1,
    ) -> "TextIndex":
        """
        Build the TextIndex . Use as follows:

        .. code-block:: python

            ix = TextIndex.build(ids, getter=getter, words_model=wv)


        Check the test cases for a better example.


        :param ids: a list of the original ids. This id's will be mapped with the internal ids used by Annoy.
        :type ids: List[Any]
        :param getter: A function which get's an ID and returns a texts to be encoded and indexed.
        :type getter: Callable
        :param words_model_path: A fulpath to the :class:`datawords.words.Word2VecHelper`
        :param words_model: optionally a :class:`datawords.words.Word2VecHelper` could be provided.
        :type words_model: Word2VecHelper
        :param vector_size: size of the vector to be indexed, it should match with vector_size of the word2vec model.
        :param distance_metric: type of distance to be used: "angular", "euclidean", "manhattan", "hamming", or "dot".
        :type distance_metric: str
        :param ann_trees: builds a forest of n_trees trees. More trees gives higher precision when querying
        :type ann_trees: int
        :param n_jobs: specifies the number of threads used to build the trees. n_jobs=-1 uses all available CPU cores.
        :type n_jobs: int
        :return: TextIndex trained.
        :rtype: TextIndex


        """
        if not words_model:
            words_model = Word2VecHelper.load(words_model_path, keyed_vectors=True)
        ix = AnnoyIndex(words_model.vector_size, distance_metric)
        id_mapper = {}
        for _ix, _id in enumerate(ids):
            data = getter(_id)
            v = words_model.encode(data)
            ix.add_item(_ix, v)
            id_mapper[_ix] = _id
        ix.build(ann_trees, n_jobs=n_jobs)
        obj = cls(
            words_model=words_model,
            id_mapper=id_mapper,
            ann_trees=ann_trees,
            distance_metric=distance_metric,
            ix=ix,
        )
        return obj

    @classmethod
    def load(
        cls, fp: Union[str, os.PathLike], words_model: Word2VecHelper = None
    ) -> "TextIndex":
        """ loads the TextIndex model.

        :param fp: The path to the index. Each model is stored in a folder.
        The path should be to that folder.
        :type fp: Union[str, os.PathLike]
        :param words_model: Optional, the words model to be used.
        :type words_model: Word2VecHelper

        :return: TextIndex loaded.
        :rtype: TextIndex
        """
        name = str(fp).rsplit("/", maxsplit=1)[1]
        with open(f"{fp}/{name}.json", "r") as f:
            jmeta = json.loads(f.read())
            meta = TextIndexMeta(**jmeta)
        ix = AnnoyIndex(meta.vector_size, meta.distance_metric)
        ix.load(f"{fp}/{name}.ann")

        with open(f"{fp}/{name}.map.json", "r") as f:
            id_mapper = json.loads(f.read())

        if not words_model:
            words_model = Word2VecHelper.load(meta.words_model_path)

        obj = cls(
            words_model=words_model,
            id_mapper=id_mapper,
            ann_trees=meta.ann_trees,
            distance_metric=meta.distance_metric,
            ix=ix,
        )
        return obj

    def save(self, fp: Union[str, os.PathLike]):
        """
        Save index to a folder.

        :param fp: The path to the index. Each model is stored in a folder.
        The path should be to that folder.
        :type fp: Union[str, os.PathLike]
        """
        name = str(fp).rsplit("/", maxsplit=1)[1]
        utils.mkdir_p(fp)
        self.ix.save(f"{fp}/{name}.ann")
        conf = TextIndexMeta(
            name=name,
            words_model_path=self.words.loaded_from,
            vector_size=self._vector_size,
            ann_tress=self._ann_trees,
            distance_metric=self._distance_metric,
        )

        with open(f"{fp}/{name}.json", "w") as f:
            f.write(conf.json())

        with open(f"{fp}/{name}.map.json", "w") as f:
            f.write(json.dumps(self.id_mapper))
