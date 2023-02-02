from typing import List

import numpy as np
import torch

# import torch.multiprocessing as mp
import torch.nn.functional as F
# from dataproc.words.utils import locale
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertModel,
    BertTokenizerFast,
    MarianMTModel,
    MarianTokenizer,
    pipeline,
)

# from toolz import partition_all
# from torch.multiprocessing import Pool, Process, set_start_method


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
       - Translate from ROMANCE langs to EN and viceversa.
       - Gets embeddings for similarity and clustering tasks.
       - Classify texts based on labels using Zero-shot technique.
       - Entity extraction using spacy library.
    """

    def __init__(self, models_path: str):
        # self.locale = locale[locale_lang]
        self.translator_tkn_romance = None
        self.translator_romance = None
        self.translator_tkn_en = None
        self.translator_en = None

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

    def load_labse(self, model_name: str = "setu4993/smaller-LaBSE"):
        """ labse is a multilengual model that can get similar vector representation for
        differents langs. """
        fullpath = f"{self.models_path}/{model_name}"
        tokenizer = BertTokenizerFast.from_pretrained(fullpath)
        model = BertModel.from_pretrained(fullpath)
        model = model.eval()
        self.labse = model
        self.labse_tokenizer = tokenizer

    def get_vectors(self, sentences, l2_norm=False):
        inputs = self.labse_tokenizer(
            sentences, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.labse(**inputs)

        embeddings = outputs.pooler_output
        if l2_norm:
            embeddings = F.normalize(embeddings, p=2)
        return embeddings

    def load_nli_model(self, model_name = "cross-encoder/nli-distilroberta-base"):
        fullpath = f"""{self.models_path}/{model_name}"""
        self.nli_tokenizer = AutoTokenizer.from_pretrained(fullpath)
        self.nli_model = AutoModel.from_pretrained(fullpath)

    def load_classifier(self, model_name = "cross-encoder/nli-distilroberta-base"):
        fullpath = f"""{self.models_path}/{model_name}"""

        self.classifier = pipeline("zero-shot-classification", model=fullpath)
        # tokenizer=self.nli_tokenizer)

    def load_features_extractor(self, model_name = "cross-encoder/nli-distilroberta-base"):
        if not self.nli_model:
            self.load_nli_model(model_name=model_name)

        self.features_extractor = pipeline("feature-extraction", model=self.nli_model,
                                           tokenizer=self.nli_tokenizer)

    def load_translate_romance_en(self, model_name="Helsinki-NLP/opus-mt-ROMANCE-en"):
        """
        print(tokenizer.supported_language_codes)
        """
        fullpath = f"""{self.models_path}/{model_name}"""
        self.translator_tkn_romance = MarianTokenizer.from_pretrained(fullpath)
        self.translator_romance = MarianMTModel.from_pretrained(fullpath)

    def load_translate_en_romance(self, model_name="Helsinki-NLP/opus-mt-en-ROMANCE"):
        """
        print(tokenizer.supported_language_codes)
        """
        fullpath = f"""{self.models_path}/{model_name}"""
        self.translator_tkn_en = MarianTokenizer.from_pretrained(fullpath)
        self.translator_en = MarianMTModel.from_pretrained(fullpath)

    def translate(self, src_text, orig="en", dst="es"):
        if orig == "en":
            s, v = self.translate_from_en(src_text, dst)
        elif orig == "es" or orig == "pt" and dst == "en":
            s, v = self.translate_romance_en(src_text)
        return s, v

    def translate_romance_en(self, src_text: str):
        """ spanish/fr,etc to EN """
        translated = self.translator_romance.generate(
            **self.translator_tkn_romance(src_text, return_tensors="pt", padding=True))
        rsp = [self.translator_tkn_romance.decode(
            t, skip_special_tokens=True) for t in translated]
        return rsp, translated

    def translate_from_en(self, src_text: str, lang: str):

        # to = [f">>{lang}<< {t}" for t in src_text]
        to = f">>{lang}<< {src_text}"
        translated = self.translator_en.generate(
            **self.translator_tkn_en(to, return_tensors="pt", padding=True))
        rsp = [self.translator_tkn_en.decode(
            t, skip_special_tokens=True) for t in translated]
        return rsp, translated

    def get_embeddings(self, text):
        rsp = self.features_extractor(text)
        return rsp[0][0]

    def classify(self, texts: List[str], labels: List[str]):
        return self.classifier(texts, labels)
