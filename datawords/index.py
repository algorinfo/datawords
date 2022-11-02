import json
from typing import Any, Callable, Dict, Optional

from annoy import AnnoyIndex
from pydantic import BaseModel

from datawords.models import WordActor, create_word_actor


class IndexConfig(BaseModel):

    words_model: str
    phrases_model: Optional[str] = None
    lang: str = "en"
    vector_size: int = 100
    ann_trees: int = 10
    distance_metric: str = "angular"


class Index:

    def __init__(self, ix: AnnoyIndex,
                 *,
                 words_model: WordActor,
                 id_mapper: Dict[int, Any],
                 conf: IndexConfig,
                 ):
        self.ix = ix
        self.id_mapper = id_mapper
        self.words = words_model
        self.conf = conf

    def encode(self, txt):
        return self.words.encode(txt)

    def save(self, fpath: str):
        self.ix.save(f"{fpath}.ann")
        with open(f"{fpath}.conf.json", "w") as f:
            f.write(self.conf.json())

        with open(f"{fpath}.maps.json", "w") as f:
            f.write(json.dumps(self.id_mapper))

    def search(self, txt, top_n=5, include_distances=False):
        v = self.encode(txt)
        vectors = self.ix.get_nns_by_vector(
            v, top_n, include_distances=include_distances)
        if include_distances:
            _res = [self.id_mapper[_v] for _v in vectors[0]]
            res = list(zip(_res, vectors[1]))
        else:
            res = [self.id_mapper[_v] for _v in vectors]
        return res


def load(bpath: str, *, index_name,
         words_model: Optional[WordActor] = None) -> Index:
    with open(f"{bpath}/models/{index_name}.conf.json", "r") as f:
        data = f.read()
        jdata = json.loads(data)
        conf = IndexConfig(**jdata)
    if not words_model:
        words_model = create_word_actor(
            bpath,
            wv_model=conf.words_model,
            phrases_model=conf.phrases_model,
            lang=conf.lang,
        )
    with open(f"{bpath}/models/{index_name}.maps.json", "r") as f:
        _ids = json.loads(f.read())
        ids_mapped = {int(k): v for k, v in _ids.items()}

    t = AnnoyIndex(conf.vector_size, conf.distance_metric)
    t.load(f"{bpath}/models/{index_name}.ann")
    ix = Index(
        t, words_model=words_model,
        id_mapper=ids_mapped,
        conf=conf,
    )
    return ix


def build_index(
        base_path: str,
        *,
        words_model: Optional[WordActor] = None,
        words_model_name: str,
        words_phrases: str = None,
        lang: str = "en",
        data_ids,
        data_getter: Callable,
        vector_size: int = 100,
        ann_trees=10,
        distance="angular"
) -> Index:
    if not words_model:
        words_model = create_word_actor(base_path,
                                        wv_model=words_model_name,
                                        phrases_model=words_phrases,
                                        lang=lang
                                        )
    f = vector_size
    t = AnnoyIndex(f, distance)
    id_mapper = {}
    for _ix, _id in enumerate(data_ids):
        data = data_getter(_id)
        v = words_model.encode(data)
        t.add_item(_ix, v)
        id_mapper[_ix] = _id

    t.build(ann_trees)
    ix = Index(t, words_model=words_model,
               id_mapper=id_mapper,
               conf=IndexConfig(
                   words_model=words_model_name,
                   phrases_model=words_phrases,
                   lang=lang,
                   vector_size=vector_size, ann_trees=ann_trees,
                   distance_metric=distance
               )
               )
    return ix
