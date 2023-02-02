import multiprocessing as mp_py
import os
from typing import List

import torch.multiprocessing as mp
# from dataproc.conf import Config
from datawords.transformers.core import Transformer
from toolz import partition_all
from tqdm import tqdm

ROMANCE_SUPPORT = ["fr", "fr_BE", "fr_CA", "fr_FR", "wa", "frp", "oc", "ca", "rm", "lld", "fur", "lij", "lmo", "es", "es_AR", "es_CL", "es_CO", "es_CR", "es_DO", "es_EC", "es_ES", "es_GT", "es_HN",
                   "es_MX", "es_NI", "es_PA", "es_PE", "es_PR", "es_SV", "es_UY", "es_VE", "pt", "pt_br", "pt_BR", "pt_PT", "gl", "lad", "an", "mwl", "it", "it_IT", "co", "nap", "scn", "vec", "sc", "ro", "la"]


class Translator:
    # pylint: disable=too-many-instance-attributes,too-few-public-methods
    LANGS = ["romance_en", "en_romance"]

    def __init__(self, orig, dst, lang_model="en_romance",
                 mp_process=True, models_path=f"models/",
                 limit_text=250, n_jobs=1):
        """ The maximum size allowed for helsinski models are 512 tokens """
        # pylint: disable=too-many-arguments

        # locale is only for compatibility reasons
        self.trf = Transformer("es-AR", models_path)
        self._func_load = getattr(self.trf, f"load_translate_{lang_model}")
        self._mp_process = mp_process
        self._orig = orig
        self._dst = dst
        self.n_jobs = self._define_njobs(n_jobs)
        # self.X = None
        # self._translated: List[str] = []
        self._limit_text = limit_text
        if not mp_process:
            self._func_load()

    @staticmethod
    def _define_njobs(n_jobs):
        if n_jobs < 0:
            return mp.cpu_count()
        else:
            return n_jobs

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

    def _process(self, X):
        # pylint: disable=import-outside-toplevel
        return_list = []
        # if not self._mp_process:
        self._func_load()

        for txt in tqdm(X):
            s, _ = self.trf.translate(
                txt[:self._limit_text], orig=self._orig, dst=self._dst)
            return_list.append(s[0])
        return return_list

    def _process_bg(self, X, return_list):
        """
        It runks in background
        """
        # pylint: disable=import-outside-toplevel
        for txt in tqdm(X):
            s, _ = self.trf.translate(
                txt[1][:self._limit_text], orig=self._orig, dst=self._dst)
            return_list.append((txt[0], s[0]))

    def fit_transform(self, X: List[str]):
        # self.X = X
        return_list: List[str] = []

        if self.n_jobs > 1:
            self.set_mp_environment()
            self._func_load()
            manager = mp.Manager()
            return_list = manager.list()
            procs = []

            _X = [(ix, x) for ix, x in enumerate(X)]

            _chunk_size = int(len(X)/self.n_jobs)
            chunks = partition_all(_chunk_size, _X)
            for chunk in chunks:
                proc = mp.Process(target=self._process_bg,
                                  args=(chunk, return_list,))
                proc.start()
                procs.append(proc)

            for p in procs:
                p.join()

            reordered_texts = sorted(return_list, key=lambda tup: tup[0])

            return [t[1] for t in reordered_texts]

        return self._process(X)


def _translate_wrapper(return_list, from_to: str, texts: List[str],
                       limit_text=200, n_jobs=1):
    result = translate_texts(from_to, texts, limit_text, n_jobs)
    return_list.extend(result)


def translate_texts_bg(from_to: str, texts: List[str],
                       limit_text=200, n_jobs=2):
    manager = mp_py.Manager()

    return_list: List[str] = manager.list()
    p = mp_py.Process(target=_translate_wrapper, args=(
        return_list, from_to, texts, limit_text, n_jobs))
    p.start()
    p.join()

    return list(return_list)


def translate_texts(from_to: str, texts: List[str], limit_text=200, n_jobs=1):
    """ It will use only one job in background. 
    :param from_to: an string with orig and source: es-en, pt_BR-en
    """
    orig, dst = from_to.split("-")

    if orig in ROMANCE_SUPPORT and dst == "en":
        lang_model = "romance_en"
    elif orig == "en" and dst in ROMANCE_SUPPORT:
        lang_model = "en_romance"

    translate = Translator(orig=orig, dst=dst,
                           lang_model=lang_model,
                           limit_text=limit_text,
                           n_jobs=n_jobs
                           )

    translated = translate.fit_transform(texts)
    return translated
