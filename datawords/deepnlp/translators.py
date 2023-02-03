import os
from typing import List

import torch.multiprocessing as mp
from toolz import partition_all
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

ROMANCE_SUPPORT = ["fr", "fr_BE", "fr_CA", "fr_FR", "wa", "frp", "oc", "ca", "rm", "lld", "fur", "lij", "lmo", "es", "es_AR", "es_CL", "es_CO", "es_CR", "es_DO", "es_EC", "es_ES", "es_GT", "es_HN",
                   "es_MX", "es_NI", "es_PA", "es_PE", "es_PR", "es_SV", "es_UY", "es_VE", "pt", "pt_br", "pt_BR", "pt_PT", "gl", "lad", "an", "mwl", "it", "it_IT", "co", "nap", "scn", "vec", "sc", "ro", "la"]


class Translator:
    """ https://huggingface.co/docs/transformers/main/model_doc/marian"""

    def __init__(self, source, target, *,
                 model_path=None,
                 limit_text=250, max_length=512):

        if model_path:
            fullpath = model_path
        else:
            fullpath = self.build_model_name(source, target)
        self._source = source
        self._target = target
        self._limit = limit_text
        self.tokenizer = MarianTokenizer.from_pretrained(
            fullpath, max_new_tokens=max_length)
        self.model = MarianMTModel.from_pretrained(fullpath)

    def flag_text_from_english(self, text: str):
        to = f">>{self._target}<< {text}"
        return to

    def _translate(self, src_text: str):
        """ spanish/fr,etc to EN """
        translated = self.model.generate(
            **self.tokenizer(src_text,
                             return_tensors="pt", padding=True))
        rsp = [self.tokenizer.decode(
            t, skip_special_tokens=True) for t in translated]
        return rsp, translated

    def transform(self, X: List[str]) -> List[str]:
        translated = []
        for ix, txt, in tqdm(enumerate(X)):
            final_txt = txt
            if self._source == "en":
                final_txt = self.flag_text_from_english(txt)
            s, _ = self._translate(final_txt[:self._limit])
            translated.append(s[0])
        return translated

    def transform_with_tensors(self, X: List[str]):
        translated = []
        tensors = []
        for ix, txt, in tqdm(enumerate(X)):
            final_txt = txt
            if self._source == "en":
                final_txt = self.flag_text_from_english(txt)
            s, t = self._translate(final_txt[:self._limit])
            translated.append(s[0])
            tensors.append(t[0])
        return translated, tensors


def _transform_wrapper(model, chunk, return_list):
    for ix_text in chunk:
        rsp = model.transform([ix_text[1]])
        return_list.append((ix_text[0], rsp[0]))


def transform_mp(source, target, *, texts,
                 n_jobs=2, model_path=None):
    """
    It uses the multiprocessing module of pytorch to work
    in multiple process
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS1"] = "1"
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_list = manager.list()
    tns = Translator(source, target, model_path=model_path) 
    procs = []
    X = [(ix, x) for ix, x in enumerate(texts)]
    _chunk_size = int(len(X)/n_jobs)
    chunks = partition_all(_chunk_size, X)
    for chunk in chunks:
        proc = mp.Process(target=_transform_wrapper,
                          args=(tns, chunk, return_list,))
        proc.start()
        procs.append(proc)

    for p in procs:
        p.join()

    reordered_texts = sorted(return_list, key=lambda tup: tup[0])

    return [t[1] for t in reordered_texts]


def build_model_name(source, target):
    _src = source
    _target = target
    if source in ROMANCE_SUPPORT:
        _src = "ROMANCE"
    elif target in ROMANCE_SUPPORT:
        _target = "ROMANCE"
    else:
        raise NameError(f"Model not found for {source} nor {target}")
    # return f"{base_path}/{model_name}"
    return f"Helsinki-NLP/opus-mt-{_src}-{_target}"
