import os
from typing import List

import torch.multiprocessing as mp
from datawords.transformers.core import Transformer
from toolz import partition_all
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertModel,
    BertTokenizerFast,
    MarianMTModel,
    MarianTokenizer,
    pipeline,
)



ROMANCE_SUPPORT = ["fr", "fr_BE", "fr_CA", "fr_FR", "wa", "frp", "oc", "ca", "rm", "lld", "fur", "lij", "lmo", "es", "es_AR", "es_CL", "es_CO", "es_CR", "es_DO", "es_EC", "es_ES", "es_GT", "es_HN",
                   "es_MX", "es_NI", "es_PA", "es_PE", "es_PR", "es_SV", "es_UY", "es_VE", "pt", "pt_br", "pt_BR", "pt_PT", "gl", "lad", "an", "mwl", "it", "it_IT", "co", "nap", "scn", "vec", "sc", "ro", "la"]


class TranslatorBase:
    """ https://huggingface.co/docs/transformers/main/model_doc/marian"""
    MODELS = {
        "romance2en": "Helsinki-NLP/opus-mt-ROMANCE-en",
        "en2romance": "Helsinki-NLP/opus-mt-en-ROMANCE"
    }

    def __init__(self, orig, dst, *, model_path="Helsinki-NLP/opus-mt-ROMANCE-en",
                 limit_text=250, max_length=512):

        self._fullpath = model_path
        self._orig = orig
        self._dst = dst
        self._limit = limit_text
        self.tokenizer = MarianTokenizer.from_pretrained(model_path, max_new_tokens=max_length)
        self.model = MarianMTModel.from_pretrained(model_path)

    @classmethod
    def get_model_fullpath(cls, base_path,*, model_name):
        return f"{base_path}/{cls.MODELS[model_name]}"

    def flag_text_from_english(self, text: str):
        to = f">>{self._dst}<< {text}"
        return to

    def _translate(self, src_text: str):
        """ spanish/fr,etc to EN """
        translated = self.model.generate(
            **self.tokenizer(src_text,
                             return_tensors="pt", padding=True))
        rsp = [self.tokenizer.decode(
            t, skip_special_tokens=True) for t in translated]
        return rsp, translated

    def fit_transform(self, X: List[str]):
        translated = []
        tensors = []
        for ix, txt, in tqdm(enumerate(X)):
            final_txt = txt
            if self._orig == "en":
                final_txt = self.flag_text_from_english(txt)
            s, t = self._translate(final_txt[:self._limit])
            translated.append(s[0])
            tensors.append(t[0])
        return translated, tensors


class Translator:
    # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """ It uses word2vec model to parse texts and get the vectors of this. """
    LANGS = ["romance_en", "en_romance"]

    def __init__(self, orig, dst, lang_model="en_romance",
                 mp_process=True, models_path=f"/models/", limit_text=250):
        """ The maximum size allowed for helsinski models are 512 tokens """

        # pylint: disable=too-many-arguments
        # self.locale_opts = locale[locale_lang]
        self.trf = Transformer("es-AR", models_path)
        self._func_load = getattr(self.trf, f"load_translate_{lang_model}")
        self._mp_process = mp_process
        self._orig = orig
        self._dst = dst
        self._limit_text = limit_text
        if not mp_process:
            self._func_load()

    @staticmethod
    def set_mp_environment():
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS1"] = "1"
        mp.set_start_method('spawn', force=True)

    @classmethod
    def get_langs_models(cls):
        return cls.LANGS

    @staticmethod
    def get_locales():
        return locale.keys()

    def _process(self, X, return_list):
        # pylint: disable=import-outside-toplevel
        if self._mp_process:
            self._func_load()
        for ix, txt in tqdm(enumerate(X)):
            if txt:
                if self._orig == "en":
                    s, _ = self.translate_from_en(
                        txt[:self._limit_text], self._dst)
                elif self._orig == "es" or orig == "pt" and dst == "en":
                    s, _ = self.translate_romance_en(
                        txt[:self._limit_text], self._dst)

                return_list.append((ix, s[0]))
            else:
                return_list.append((ix, ""))

    def fit_transform(self, X: List[str]):
        # self.X = X

        return_list: List[str] = []
        if self._mp_process:
            self.set_mp_environment()
            manager = mp.Manager()
            return_list = manager.list()
            proc = mp.Process(target=self._process, args=(X, return_list))
            proc.start()
            proc.join()
            return list(return_list)

        self._process(X, return_list)
        return self._translated


def wrapper_fit_transform(model, texts, return_list):
    """ If the text is to long it will fail """
    try:
        texts_en = model.fit_transform(texts)
    except IndexError:
        texts_en = [""]
    return_list.extend(texts_en)


def translate_texts_en(orig_lang, titles, desc, return_dict, limit_text=200):
    manager = mp.Manager()
    titles_en = manager.list()
    desc_en = manager.list()
    translate = Translator(orig=orig_lang, dst="en",
                           lang_model="romance_en", mp_process=False,
                           limit_text=limit_text)
    translate.set_mp_environment()
    proc_t = mp.Process(target=wrapper_fit_transform,
                        args=(translate, titles, titles_en))
    proc_d = mp.Process(target=wrapper_fit_transform,
                        args=(translate, desc, desc_en))
    proc_t.start()
    proc_d.start()
    proc_t.join()
    proc_d.join()
    return_dict["titles"] = list(titles_en)
    return_dict["desc"] = list(desc_en)


def translate_texts(orig: str, *, dst: str, texts: List[str], limit_text=200):
    """ It will use only one job in background.
    :param from_to: an string with orig and source: es-en, pt_BR-en
    """
    # orig, dst = from_to.split("-")

    if orig in ROMANCE_SUPPORT and dst == "en":
        lang_model = "romance_en"
    elif orig == "en" and dst in ROMANCE_SUPPORT:
        lang_model = "en_romance"

    translate = Translator(orig=orig, dst=dst,
                           lang_model=lang_model, mp_process=True,
                           limit_text=limit_text)

    translated = translate.fit_transform(texts)
    return translated
