import sqlite3
import warnings
from collections import Counter
from typing import Dict, List

import numpy as np
from attrs import define
from sklearn.metrics.pairwise import cosine_similarity

from datawords import parsers


@define
class Entity:
    txt: str
    start_char: int
    end_char: int
    label: str


def extract_entities(doc) -> List[Entity]:
    """Gets a spacy doc object"""
    _entities = []
    for ent in doc.ents:
        e = Entity(ent.text, ent.start_char, ent.end_char, ent.label_)
        _entities.append(e)
    return _entities


def entities_by_label(entities: List[Entity]) -> Dict[str, List[str]]:
    by_label = {}
    for e in entities:
        if by_label.get(e.label):
            by_label[e.label].append(e.txt)
        else:
            by_label.update({e.label: [e.txt]})

    return by_label


def _run_query(cursor: sqlite3.Cursor, q: str):
    """sqlite helper to run queries."""
    return cursor.execute(q).fetchall()


def _longest(data):
    """
    Helper to choose the string with more words:
    [('Maria Perez',), ('Maria Perez de Gozales',), ('Maria',)]
    it will return the second one.
    """
    aux = 0
    final = None
    for x in data:
        if len(x[0].split()) > aux:
            final = x
            aux = len(x)
    return final


def compute_similarity_m(vects):
    """
    It gets a cosine similarity matrix
    the diagonal where each element match with him self is
    filled with 0 values. if values are not zeros
    then pagerank will fail
    """
    A = np.array(vects)
    # A_sparse = sparse.csr_matrix(A)
    M = cosine_similarity(A)

    for i in range(len(vects)):
        for j in range(len(vects)):
            if i == j:
                M[i][j] = 0.0
    return M


def norm_l2_np(vec):
    """
    Normalize a vector. Similar to pytorch norm.
    """
    return vec / np.linalg.norm(vec)


def norm_l2_np_with_zeros(vec):
    """
    If a vector is all zeros it will avoid the normalization
    """
    if not vec.sum() == 0.0:
        return vec / np.linalg.norm(vec)
    return vec


class WordSearch:
    def __init__(self, sqlite=":memory:", stopwords=set()):
        """
        Based  on a list we want to get uniques names ready to be used
        with fts5 type data from sqlite3 which allow us to perform fuzzy
        search over a set of data.
        """
        self.db = sqlite3.connect(sqlite)
        self._create_tables()
        self._stopw = stopwords

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

    @staticmethod
    def _run_query(cursor: sqlite3.Cursor, q: str):
        """sqlite helper to run queries."""
        return cursor.execute(q).fetchall()

    def _create_tables(self):
        cur = self.db.cursor()
        cur.execute('create virtual table vtags using fts5(name, tokenize="ascii");')
        cur.execute(
            'create virtual table unique_words using fts5(name, tokenize="ascii");'
        )
        cur.execute(
            """CREATE TABLE normal
        (id INTEGER PRIMARY KEY,name TEXT NOT NULL UNIQUE);"""
        )
        cur.close()

    def list_words(self, table="vtags") -> List[str]:
        cur = self.db.cursor()
        rows = cur.execute(f"select * from {table};")

        return [r[0] for r in rows]

    def add(self, word: str) -> bool:
        cur = self.db.cursor()
        added = False
        try:
            cur.execute("insert into normal (name) values (?);", (word,))
            cur.execute("insert into vtags (name) values (?);", (word,))
            self.db.commit()
            added = True
        except sqlite3.IntegrityError:
            pass
        except sqlite3.OperationalError:
            pass

        cur.close()
        return added

    def add_batch(self, words: List[str]) -> List[bool]:
        cur = self.db.cursor()
        tracking = []
        for w in words:
            try:
                cur.execute("insert into normal (name) values (?);", (w,))
                cur.execute("insert into vtags(name) values(?);", (w,))
                tracking.append(True)
            except sqlite3.IntegrityError:
                tracking.append(False)
        self.db.commit()
        cur.close()
        return tracking

    def get(self, word: str, limit: int = 1) -> List[str]:
        warnings.warn("WordsSearch.get() will be deprecated", DeprecationWarning)
        return self.fuzzy_search(word, limit)

    def fuzzy_search(self, name: str, limit: int = 1) -> List[str]:
        cur = self.db.cursor()
        try:
            result = self._run_query(
                cur,
                f"""select * from vtags where name MATCH '"{name}" *'
                                    limit {limit}""",
            )
        except sqlite3.OperationalError:
            result = []
        cur.close()
        return [r for r, in result]

    def ngram_search(self, txt: str, ngrams=2, table="vtags") -> Counter:
        """
        It's an approximate counter of repeated words in a text.

        It counts how often each word in the corpus appears in a table.
        Based on the names previously loaded with `get_uniques`
        this method receive a text and counts how often each name appears
        """
        cur = self.db.cursor()
        c = Counter()
        tokens = self._parse(txt)
        ngrams = parsers.generate_ngrams(tokens, n=ngrams)
        for x in ngrams:
            result = _run_query(
                cur,
                f"""select * from {table} where name MATCH '"{x}" *'
                        limit 1""",
            )
            if len(result) > 0:
                c.update(result[0])

        return c
