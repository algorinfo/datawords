import os
from typing import Any, Callable, Dict, List, Optional

import torch.multiprocessing as mp
from dataproc.conf import Config
from dataproc.words.transformers.core import Transformer
from dataproc.words.utils import locale
# from tqdm import tqdm


def default_filter_classifier(data):
    return data["labels"][0], data["scores"][0]


class Classifier:
    # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """ A wrapper around the pytorch transformer. It perform zero-shot predictions of texts
    based on a list of labels provided."""

    def __init__(self, locale_lang,
                 mp_process=True, models_path=f"{Config.BASE_PATH}/models/",
                 filter_response: Optional[Callable] = default_filter_classifier):
        # pylint: disable=too-many-arguments
        # self.locale_opts = locale[locale_lang]
        """
        :param locale_lang: a ISO locale should be provided, it will find the model by locale.
        :param mp_process: if True, it will load the model and make the predictions 
        in a background process
        :param models_path: path where to look for models.
        :param filter_response: a callable to filter the prediction response from the model.
        """
        self.trf = Transformer(locale_lang, models_path)
        self._func_load = getattr(self.trf, "load_classifier")
        self._mp_process = mp_process
        self.X: List[str] = []
        self._predicted: List[Dict[str, Any]] = []
        if not mp_process:
            self._func_load()
        self._filter_response: Optional[Callable] = filter_response

    @staticmethod
    def set_mp_environment():
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS1"] = "1"
        mp.set_start_method('spawn', force=True)

    @staticmethod
    def get_locales():
        return locale.keys()

    def _process(self, return_list, labels):
        # pylint: disable=import-outside-toplevel
        if self._mp_process:
            self._func_load()
        # for txt in tqdm(self.X):
        for txt in self.X:
            prediction = self.trf.classifier([txt], labels)
            if self._filter_response:
                return_list.append(self._filter_response(prediction[0]))
            else:
                return_list.append(prediction[0])

    def fit_transform(self, X: List[str], labels: List[str]):
        self.X = X
        return_list: List[Dict[str, Any]] = []
        if self._mp_process:
            self.set_mp_environment()
            manager = mp.Manager()
            return_list = manager.list()
            proc = mp.Process(target=self._process, args=(return_list, labels))
            proc.start()
            proc.join()
            self._predicted = list(return_list)
            return self._predicted

        self._process(return_list, labels)
        self._predicted = return_list
        return self._predicted
