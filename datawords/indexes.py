import json
import os
import sqlite3
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from annoy import AnnoyIndex
from attrs import asdict, define
from tqdm import tqdm

from datawords import _utils, parsers
from datawords.models import Word2VecHelper


@define
class LiteDoc:
    """
    Represents a document indexed in the :class:`SQLiteIndex`
    """

    id: str
    text: str

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


@define
class TextIndexMeta:
    name: str
    words_model_path: str
    vector_size: int = 100
    ann_trees: int = 10
    distance_metric: str = "angular"
    version: str = _utils.get_version()


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
        This Index is opinated, usually AnnoyIndex allows any kind of Vector,
        insted, this index is prepared to accept only texts, parse it and
        perform searchs.

        AnnoyIndex only accepts integers as indices of the data,
        for that reason and id_mapper is built in the build process.


        :param words_model: Words Model to be used for vectorizing texts.
        :type words_model: Word2VecHelper
        :param id_mapper: A dict where keys are the indices stored in Annoy
            and values are the real indices in the domain which the data
            belongs to.
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
        ids: Iterable,
        *,
        getter: Callable,
        words_model_path: Optional[str] = None,
        words_model: Optional[Word2VecHelper] = None,
        distance_metric: str = "angular",
        ann_trees: int = 10,
        n_jobs: int = -1,
        progress_bar=True,
        total_ids=None,
    ) -> "TextIndex":
        """
        Build the TextIndex . Use as follows:

        .. code-block:: python

            def getter(id_: str) -> str:
                return db.get(id_)

            ix = TextIndex.build(ids, getter=getter, words_model=wv)


        Check the test cases for a better example.


        :param ids: a list of the original ids. This id's will be mapped
            with the internal ids used by Annoy.
        :type ids: List[Any]
        :param getter: A function which get's an ID and returns a texts
            to be encoded and indexed.
        :type getter: Callable
        :param words_model_path: A fulpath to the
            :class:`datawords.words.Word2VecHelper`
        :param words_model: optionally a
            :class:`datawords.words.Word2VecHelper` could be provided.
        :type words_model: Word2VecHelper
        :param vector_size: size of the vector to be indexed, it should
            match with vector_size of the word2vec model.
        :param distance_metric: type of distance to be used: "angular",
            "euclidean", "manhattan", "hamming", or "dot".
        :type distance_metric: str
        :param ann_trees: builds a forest of n_trees trees. More trees
            gives higher precision when querying
        :type ann_trees: int
        :param n_jobs: specifies the number of threads used to build the
            trees. n_jobs=-1 uses all available CPU cores.
        :type n_jobs: int
        :return: TextIndex trained.
        :rtype: TextIndex

        """
        if not words_model:
            words_model = Word2VecHelper.load(words_model_path, keyed_vectors=True)
        ix = AnnoyIndex(words_model.vector_size, distance_metric)
        id_mapper = {}
        for _ix, _id in tqdm(enumerate(ids), disable=not progress_bar, total=total_ids):
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
        cls, fp: Union[str, os.PathLike], words_model: Optional[Word2VecHelper] = None
    ) -> "TextIndex":
        """loads the TextIndex model.

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
            idm = json.loads(f.read())

        id_mapper = {int(k): v for k, v in idm.items()}

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
        _utils.mkdir_p(fp)
        self.ix.save(f"{fp}/{name}.ann")
        conf = TextIndexMeta(
            name=name,
            words_model_path=self.words.loaded_from,
            vector_size=self._vector_size,
            ann_trees=self._ann_trees,
            distance_metric=self._distance_metric,
        )

        with open(f"{fp}/{name}.json", "w") as f:
            _d = asdict(conf)
            f.write(json.dumps(_d))

        with open(f"{fp}/{name}.map.json", "w") as f:
            f.write(json.dumps(self.id_mapper))


class SQLiteIndex:
    def __init__(self, sqlite=":memory:", stopwords=set()):
        """
        SQLiteIndex allows to store documents and search over them.
        It uses the fts5 module from sqlite. Also it parse the text
        in an opinated way.

        :param sqlite: path where database will be stored.
            By default it's saved on memory
        :type sqlite: str
        :param stopwords: list of stop words to be used by the parsed.
        :type stopwords: Set[str]
        """
        self.db = sqlite3.connect(sqlite)
        self._create_tables()
        self._stopw = stopwords

    def journal_mode(self):
        cur = self.db.cursor()
        cur.execute("PRAGMA jounral_mode=WAL;")
        cur.close()

    def _create_tables(self):
        cur = self.db.cursor()
        cur.execute(
            'CREATE VIRTUAL TABLE IF NOT EXISTS search_docs using fts5(id, text, tokenize="ascii");'
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS search_index
        (id INTEGER PRIMARY KEY,doc_id TEXT NOT NULL UNIQUE);"""
        )
        cur.close()

    def _parse(self, txt) -> List[str]:
        tokens = parsers.doc_parser(
            txt,
            self._stopw,
            emo_codes=False,
            strip_accents=True,
            lower=True,
            numbers=True,
            parse_urls=False,
        )
        return tokens

    def parse(self, txt) -> List[str]:
        """
        Parse a text
        """
        return self._parse(txt)

    def _insert(self, cur, doc: LiteDoc):
        tokens = self._parse(doc.text)
        words = " ".join(tokens)
        cur.execute(
            "insert into search_index (doc_id) values (?);",
            (doc.id,),
        )
        cur.execute(
            "insert into search_docs (id, text) values (?, ?);",
            (
                doc.id,
                words,
            ),
        )

    def add(self, doc: LiteDoc) -> bool:
        """
        Adds a document of type :class:`LiteDoc` into the index.
        If the document already exist, then it will not be stored.

        :param doc: A document
        :type doc: LiteDoc
        :return: True if it was stored or false if not.
        :rtype: bool
        """
        cur = self.db.cursor()
        added = False
        try:
            self._insert(cur, doc)

            self.db.commit()
            added = True
        except sqlite3.IntegrityError:
            pass
        except sqlite3.OperationalError:
            pass

        cur.close()
        return added

    @classmethod
    def build(
        cls,
        ids: Iterable,
        *,
        getter: Callable,
        stopwords=set(),
        progress_bar=True,
        total_ids=None,
        sqlite: str = ":memory:",
    ) -> "SQLiteIndex":
        obj = cls(sqlite, stopwords)
        cur = obj.db.cursor()
        tracking = []
        for _id in tqdm(ids, disable=not progress_bar, total=total_ids):
            data = getter(_id)
            doc = LiteDoc(id=_id, text=data)
            try:
                obj._insert(cur, doc)
                tracking.append(True)
            except sqlite3.IntegrityError:
                tracking.append(False)
        obj.db.commit()
        cur.close()
        return obj

    @classmethod
    def load(cls, sqlite: str, stopwords=set()) -> "SQLiteIndex":
        obj = cls(sqlite, stopwords)
        return obj

    def add_batch(self, docs: List[LiteDoc]) -> List[bool]:
        """
        Add documents in batch.

        :param docs: A list of documents
        :type docs: List[LiteDoc]
        :return: True if it was stored or false if not.
        :rtype: List[bool]
        """
        cur = self.db.cursor()
        tracking = []
        for doc in docs:
            try:
                self._insert(cur, doc)
                tracking.append(True)
            except sqlite3.IntegrityError:
                tracking.append(False)
        self.db.commit()
        cur.close()
        return tracking

    def _list(self, cur, limit=10, offset=0, table="search_index"):
        result = cur.execute(f"select * from {table} LIMIT {offset}, {limit};")
        return result

    def get_doc(self, id: str) -> LiteDoc:
        cur = self.db.cursor()
        # row = cur.execute(f"select * from search_docs where search_docs.id={id};").fetchone()

        row = cur.execute(
                f"""select * from search_docs where text MATCH '{id}'
                                    limit 1"""
        ).fetchone()

        cur.close()
        return LiteDoc(id=row[0], text=row[1])

    def list_docs(self, limit=10, offset=0) -> List[LiteDoc]:
        cur = self.db.cursor()
        rows = self._list(cur, limit=limit, offset=offset, table="search_docs")
        docs = [LiteDoc(id=r[0], text=r[1]) for r in rows]
        cur.close()
        return docs

    def list_ids(self, limit=10, offset=0) -> List[str]:
        """
        It lists the ids of the document stored.
        """
        cur = self.db.cursor()
        rows = self._list(cur, limit=limit, offset=offset, table="search_index")
        ids = [r[1] for r in rows]
        cur.close()
        return ids

    @property
    def total(self) -> int:
        """Returns the total of documents stored in the index."""
        cur = self.db.cursor()
        res = cur.execute("select count(*) from search_index;").fetchone()
        cur.close()
        return res[0]

    def search(self, text: str, top_n: int = 5) -> List[LiteDoc]:
        """
        Performs a search in the index. It will parse and match
        the wodos as:

        MATCH '"{words}" *'

        To use a more low level search query use
        :method:`SQLiteIndex.search_query`


        :param text: text to search.
        :type text: str
        :param limit: how many results retrieve.
        :type limit: int
        :return: Documents found
        :rtype: List[LiteDoc]
        """
        tokens = self._parse(text)
        words = " ".join(tokens)
        cur = self.db.cursor()
        try:
            result = cur.execute(
                f"""select * from search_docs where text MATCH '"{words}" *'
                                    limit {top_n}"""
            ).fetchall()

        except sqlite3.OperationalError:
            result = []
        cur.close()
        return [LiteDoc(id=r[0], text=r[1]) for r in result]

    def search_query(self, query: str, top_n: int = 5) -> List[LiteDoc]:
        """
        Performs a search query into the sqlite database.

        'search AND (sqlite OR help)'

        :param query: query to search
        :type query: str
        :param limit: how many results retrieve.
        :type limit: int
        :return: Documents found
        :rtype: List[LiteDoc]
        """
        # tokens = self._parse(text)
        cur = self.db.cursor()
        try:
            result = cur.execute(
                f"""select * from search_docs where text MATCH '{query}'
                                    limit {top_n}"""
            ).fetchall()

        except sqlite3.OperationalError:
            result = []
        cur.close()
        return [LiteDoc(id=r[0], text=r[1]) for r in result]
