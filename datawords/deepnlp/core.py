from typing import List, Optional

import numpy as np
import torch

import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertModel,
    BertTokenizerFast,
    pipeline,
)

def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def norm_l2_torch(vect):
    norm = F.normalize(vect, p=2)
    return norm


def to_np_arr(vect, l2_norm=True):
    if l2_norm:
        norm = norm_l2_torch(vect)
        return norm.detach.numpy()
    return vect.detach.numpy()


class Transformer:
    """ A Bag for differents NLP tasks using transformers from
    HuggingFace and Spacy.
    Tasks:
       - Gets embeddings for similarity and clustering tasks.
       - Classify texts based on labels using Zero-shot technique.
       - Entity extraction using spacy library.
    """

    def __init__(self, models_path: Optional[str]=None):
        # classifier and encoder based on NLI Distil Roberta
        self.nli_model = None
        self.nli_tokenizer = None
        self.classifier = None
        self.features_extractor = None

        self.models_path = models_path
        self.nlp = None
        self.nlp_en = None

        self.labse = None
        self.labse_tokenizer = None

    def _get_fullpath(self, model_name: str) -> str:
        fp = model_name
        if self.models_path:
            fp = f"{self.models_path}/{model_name}"
        return fp

    def load_labse(self, model_name: str = "setu4993/smaller-LaBSE"):
        """ labse is a multilengual model that can get similar vector representation for
        differents langs. """
        fullpath = self._get_fullpath(model_name)
        tokenizer = BertTokenizerFast.from_pretrained(fullpath)
        model = BertModel.from_pretrained(fullpath)
        model = model.eval()
        self.labse = model
        self.labse_tokenizer = tokenizer

    def multilingual_vectors(self, sentences, l2_norm=False):
        inputs = self.labse_tokenizer(
            sentences, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.labse(**inputs)

        embeddings = outputs.pooler_output
        if l2_norm:
            embeddings = F.normalize(embeddings, p=2)
        return embeddings

    def load_nli_model(self,
                       model_name="cross-encoder/nli-distilroberta-base"):
        fullpath = self._get_fullpath(model_name)
        self.nli_tokenizer = AutoTokenizer.from_pretrained(fullpath)
        self.nli_model = AutoModel.from_pretrained(fullpath)

    def load_classifier(self,
                        model_name="cross-encoder/nli-distilroberta-base"):
        fullpath = self._get_fullpath(model_name)

        self.classifier = pipeline("zero-shot-classification", model=fullpath)
        # tokenizer=self.nli_tokenizer)

    def load_features_extractor(self,
                                model_name="cross-encoder/nli-distilroberta-base"):
        if not self.nli_model:
            self.load_nli_model(model_name=model_name)

        self.features_extractor = pipeline("feature-extraction", model=self.nli_model,
                                           tokenizer=self.nli_tokenizer)

    def get_embeddings(self, text):
        rsp = self.features_extractor(text)
        return rsp[0][0]

    def classify(self, texts: List[str], labels: List[str]):
        return self.classifier(texts, labels)


