import json
import os
from typing import Any, Callable, Dict, List, Union

from annoy import AnnoyIndex
from pydantic import BaseModel

from datawords import utils
from datawords.models import Word2VecHelper


class WordsIndexMeta(BaseModel):
    name: str
    words_model_path: str
    vector_size: int = 100
    ann_trees: int = 10
    distance_metric: str = "angular"
    version: str = utils.get_version()


class WordsIndex:
    def __init__(
        self,
        words_model: Word2VecHelper,
        id_mapper: Dict[int, Any],
        ann_trees: int = 10,
        distance_metric: str = "angular",
        ix: AnnoyIndex = None,
    ):
        self.ix: AnnoyIndex = ix
        self.id_mapper = id_mapper
        self.words = words_model
        self._ann_trees = ann_trees
        self._distance_metric = distance_metric
        self._vector_size = words_model.vector_size

    def encode(self, txt: str):
        return self.words.encode(txt)

    def search(self, txt, top_n=5, include_distances=False):
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
        vector_size: int = 100,
        distance_metric: str = "angular",
        ann_trees: int = 10,
    ) -> "WordsIndex":
        if not words_model:
            words_model = Word2VecHelper.load(words_model_path,
                                              keyed_vectors=True)
        ix = AnnoyIndex(vector_size, distance_metric)
        id_mapper = {}
        for _ix, _id in enumerate(ids):
            data = getter(_id)
            v = words_model.encode(data)
            ix.add_item(_ix, v)
            id_mapper[_ix] = _id
        ix.build(ann_trees)
        obj = cls(
            words_model=words_model,
            id_mapper=id_mapper,
            ann_trees=ann_trees,
            distance_metric=distance_metric,
            ix=ix
        )
        return obj

    @classmethod
    def load(
        cls, fp: Union[str, os.PathLike], words_model: Word2VecHelper = None
    ) -> "WordsIndex":
        name = str(fp).rsplit("/", maxsplit=1)[1]
        with open(f"{fp}/{name}.json", "r") as f:
            jmeta = json.loads(f.read())
            meta = WordsIndexMeta(**jmeta)
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
        name = str(fp).rsplit("/", maxsplit=1)[1]
        utils.mkdir_p(fp)
        self.ix.save(f"{fp}/{name}.ann")
        conf = WordsIndexMeta(
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
