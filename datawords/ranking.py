# from scipy import sparse
from typing import Any, List

import numpy as np
from annoy import AnnoyIndex
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sknetwork.ranking import PageRank
from tqdm import tqdm

from datawords.utils import norm_l2_np_with_zeros


def get_adj(edges, size):
    E = edges
    # size = max(max(E))+1
    r = [[0 for i in range(size)] for j in range(size)]
    for row, col in E:
        r[row][col] = 1
    return np.asarray(r)


def build_annoy_edges(ix: AnnoyIndex, vectors: np.ndarray, barrier=0.9, tqdm_=True):
    edges = []
    if tqdm_:
        for i in tqdm(range(vectors.shape[0])):
            for j in range(vectors.shape[0]):
                if i != j:
                    dst = ix.get_distance(i, j)
                    if dst <= barrier:
                        edges.append((i, j))
    else:
        for i in range(vectors.shape[0]):
            for j in range(vectors.shape[0]):
                if i != j:
                    dst = ix.get_distance(i, j)
                    if dst <= barrier:
                        edges.append((i, j))

    return edges


class PageRankTFIDF:
    # pylint: disable=too-few-public-methods
    """PageRank based on TFIDF"""

    def __init__(
        self, tokenizer=None, barrier=0.7, min_df=1, max_df=0.9, ngram_range=(1, 1)
    ):
        # pylint: disable=too-many-arguments
        self.tf = TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            # stop_words=stopw,
            tokenizer=tokenizer,
            min_df=min_df,
            max_df=max_df,
        )
        self.barrier = barrier

    @staticmethod
    def _compute_tfidf_M(A):
        M = cosine_similarity(A)

        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                if i == j:
                    M[i][j] = 0.0
        return M

    @staticmethod
    def _build_edges_all(A, barrier=0.7):
        edges = []
        M = cosine_similarity(A)

        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                if i != j:
                    cosine = M[i, j]
                    if cosine > barrier:
                        # edges.append((i, j, cosine))
                        edges.append((i, j))

        return edges

    def fit_transform(self, corpus):
        """
        Estimate a pagerank by cosine_similarity
        """
        tf_M = self.tf.fit_transform(corpus)
        sim_M = self._compute_tfidf_M(tf_M)
        adjacency = sparse.csr_matrix(sim_M)
        pagerank = PageRank()
        scores = pagerank.fit_transform(adjacency)
        return scores


# corpus: Union[Generator, List[str]]
class PageRankAnnoy:
    # pylint: disable=too-many-instance-attributes
    """
    It uses Annoy Indexer and PageRank to rank a list of `documents`,
    where each document should be an string.

    PageRank is an old known google algorithm, originally used to get the
    "most important" links from internet.

    In the text world, there are different approachs to achive the same goal.
    For instance, if two documents share the same words, then they could
    be considered as nodes connected between.

    Because PageRank is a graph algorithm the idea of nodes
    and edges is important.

    Other approach is to measure the similarity between
    two documents (nodes), if they are similar enough (`barrier`),
    then they are connected. This last method is used here.

    """

    def __init__(
        self,
        metric_distance="euclidean",
        l2_norm=True,
        barrier=0.9,
        n_trees=10,
        n_jobs=-1,
        tqdm=True,
    ):
        """
        For `l2_norm` param refer to
        https://machinelearningmastery.com/vector-norms-machine-learning/

        :param metric_distance: the options are  "angular", "euclidean",
            "manhattan", "hamming", or "dot". It will be used by Annoy
             as measure to calculates distance between texts.

        :param l2_norm: True by default, usually is recommend their usage.
        :param barrier: It depends on the metric_distance choose, but
            this param will serve as filter to define if 2 texts are
            connected as nodes.
        :param n_trees: trees used by annoy index.
        :param n_jobs: using multithreading for the index.
        :param tqdm: tqdm usage when adj. matrix is calculated.

        """
        self.barrier = barrier
        self.metric_distance = metric_distance
        self.n_trees = n_trees
        self.n_jobs = n_jobs
        self.X = None
        self.aix = None
        self._adj = None
        self._edges = None
        self._scores = None
        self._tqdm = tqdm
        self._l2_norm = l2_norm
        # self.annoy_ix = AnnoyIndex(vector_size, metric_distance)
        # self.vectors = None

    @staticmethod
    def as_ndarray(X: List[np.ndarray]) -> np.ndarray:
        return np.asarray(X)

    def fit(self, X: np.ndarray):
        self.X = X
        if self._l2_norm:
            self.X = np.apply_along_axis(norm_l2_np_with_zeros, 1, X)

    def transform(self):
        self.aix = AnnoyIndex(self.X.shape[1], self.metric_distance)
        for i in range(self.X.shape[0]):
            self.aix.add_item(i, self.X[i])

        self.aix.build(n_trees=self.n_trees, n_jobs=self.n_jobs)
        pg = PageRank()
        self._edges = build_annoy_edges(
            self.aix, self.X, barrier=self.barrier, tqdm_=self._tqdm
        )
        self._adj = get_adj(self._edges, self.X.shape[0])

        self._scores = pg.fit_transform(self._adj)
        return self._scores

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        scores = self.transform()
        return scores

    def rank(self, index: List[Any], top_n=5):
        """given a `index` build a ranking using the scores created
        by PageRank."""
        ranking = zip(index, self.scores)
        best = sorted(ranking, key=lambda tup: tup[1], reverse=True)[:top_n]
        return best

    @property
    def scores(self):
        return self._scores

    @property
    def edges(self):
        return self._edges

    @property
    def adjacency(self):
        return self._adj


def rank(index: List[Any], scores: np.ndarray, top_n=5):
    """
    given a score result from any of the PageRank models
    an a index which correlates with the data used to calculate
    pagerank then, reorder score from top to bottom.

    To produce a simple numeric index, :function:`datawords.ranking.numeric_index`
    could be used. 
    """
    ranking = zip(index, scores)
    best = sorted(ranking, key=lambda tup: tup[1], reverse=True)
    if top_n:
        return best[:top_n]
    return best


def numeric_index(values: List[Any]) -> List[int]:
    """
    given a list of values of any type it produce
    an incremental index of int.
    useful for :function:`datawords.ranking.rank`
    """
    return list(range(0, len(values)))
