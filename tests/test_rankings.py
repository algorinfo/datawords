import numpy as np

from datawords import models, parsers, ranking
from .shared import open_texts


def test_ranking_pagerank_annoy():
    texts = list(open_texts())
    parser_conf = parsers.ParserConf(lang="en")
    wv = models.Word2VecHelper(parser_conf)
    wv.fit(texts)
    vects = [wv.encode(t) for t in texts]

    pra = ranking.PageRankAnnoy()
    arrs = pra.as_ndarray(vects)
    scores = pra.fit_transform(arrs)
    index = [ix for ix, _ in enumerate(texts)]
    rank = pra.rank(index)
    assert len(rank) == 5
    assert isinstance(scores, np.ndarray)


def test_ranking_tfidf():
    texts = list(open_texts())
    pr = ranking.PageRankTFIDF()
    scores = pr.fit_transform(texts)
    index = ranking.numeric_index(texts)
    rank = ranking.rank(index, scores)
    assert len(rank) == 5
    assert isinstance(scores, np.ndarray)
