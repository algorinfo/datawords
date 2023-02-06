from typing import List

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.phrases import FrozenPhrases, Phrases
from sklearn.metrics.pairwise import cosine_similarity

# import spacy
from datawords.parsers import doc_parser, load_stop

# from sknetwork.ranking import PageRank


def compute_similarity_m(vects):
    """get a cosine similarity matrix
    the diagonal where each element match with their self is
    matched to 0, if not pagerank fail
    """
    A = np.array(vects)
    # A_sparse = sparse.csr_matrix(A)
    M = cosine_similarity(A)

    for i in range(len(vects)):
        for j in range(len(vects)):
            if i == j:
                M[i][j] = 0.0
    return M


class WordActor:

    def __init__(self, stopw, wv_model: Word2Vec, nlp=None,
                 phrases: FrozenPhrases = None):
        self.nlp = nlp
        self.stopw = stopw
        self.wv: Word2Vec = wv_model
        self.vector_size = self.wv.vector_size
        self.phrases = phrases

    def load_nlp(self, nlp):
        self.nlp = nlp

    def doc_parser(self, txt, strip_accents=True):
        return doc_parser(txt, self.stopw, strip_accents=strip_accents)

    def get_vector2(self, s: List[str]):
        """ Get a vector from a list of words
        if a sentence doesn't match, then it fills with zeros
        """
        vectors = []
        for word in s:
            try:
                _vect = self.wv[word]
                vectors.append(_vect)
            except KeyError:
                pass
        if len(vectors) > 0:
            return np.sum(np.array(vectors), axis=0) / (len(vectors) + 0.001)
        else:
            return np.zeros((self.vector_size,))

    def encode(self, txt: str):
        """
        gets a sentence in plain text and encode it as vector
        """
        words = self.doc_parser(txt)
        if self.phrases:
            sentence = self.phrases[words]
        else:
            sentence = words
        v = self.get_vector2(sentence)
        return v

    # def pagerank(self, df, column="title", strip_accents=True):
    #     """
    #     Estimate a pagerank by cosine_similarity
    #     """
    #     vects, _ = self.sentences(df, column, strip_accents)
    #     sim_M = compute_similarity_m(vects)
    #     adjacency = sparse.csr_matrix(sim_M)
    #     pagerank = PageRank()
    #     scores = pagerank.fit_transform(adjacency)
    #     return vects, scores

    def init_nlp_doc(self, text: str):
        return self.nlp(text)

    def similarity(self, txt1, txt2, strip_accents=True):
        vec1 = self.get_vector2(doc_parser(txt1,
                                           self.stopw,
                                           strip_accents=strip_accents))
        vec2 = self.get_vector2(doc_parser(txt2, self.stopw,
                                           strip_accents=strip_accents))
        simil = cosine_similarity([vec1], [vec2])
        return simil[0][0]


def create_word_actor(base_path: str,
                      wv_model: str,
                      phrases_model: str = None,
                      lang="en",
                      nlp_model_name=None,
                      nlp_rank=False) \
        -> WordActor:
    """ Helper function for a singleton word actor model """
    # pylint: disable=maybe-no-member
    # pylint: disable=
    nlp = None
    stopw = load_stop(
        f"{base_path}/models", lang=lang)

    wv_path = f"{base_path}/models/{wv_model}"
    wv = KeyedVectors.load(wv_path, mmap='r')

    if nlp_model_name:
        import spacy
        nlp = spacy.load(f"{base_path}/models/{nlp_model_name}")
        if nlp_rank:
            import pytextrank
            nlp.add_pipe("textrank")

    phrases = None
    if phrases_model:
        phrases = Phrases.load(f"{base_path}/models/{phrases_model}")
    actor = WordActor(stopw, wv_model=wv.wv, nlp=nlp, phrases=phrases)

    return actor
