from typing import List

from toolz import partition_all
from tqdm.auto import tqdm
from transformers import MarianMTModel, MarianTokenizer, pipeline

from datawords.constants import ROMANCE_SUPPORT
from datawords.deepnlp.utils import get_gpu_info


class TranslatorBeta:
    ROMANCE_SUPPORT = ROMANCE_SUPPORT

    def __init__(
        self,
        source,
        target,
        *,
        model_path=None,
        limit_text=250,
        max_length=512,
        device=-1,
        automatic_gpu_detection=True,
        progress_bar=True,
        chunk_size=50,
    ):
        """
        :param source: lang source, it could be any ROMANCE lang, or `en`.
        :type source: str
        :param target: Any ROMANCE or en lang.
        :type target: str

        """

        if model_path:
            fullpath = model_path
        else:
            fullpath = self.build_model_name(source, target)
        self._source = source
        self._target = target
        self._limit = limit_text
        self.tokenizer = MarianTokenizer.from_pretrained(
            fullpath, model_max_length=max_length, truncation=True
        )
        self.model = MarianMTModel.from_pretrained(fullpath)
        _device = self._get_device() if automatic_gpu_detection else device
        self.using_device = _device
        self.pipe = pipeline(
            f"translation_{source}_to_{target}",
            model=self.model,
            tokenizer=self.tokenizer,
            device=_device,
        )
        self._progress_bar = progress_bar
        self._chunk = chunk_size

    def _get_device(self) -> int:
        gpu = get_gpu_info()
        if gpu:
            return 0
        return -1

    def transform(self, texts: List[str]) -> List[str]:
        """
        TODO: check how to bring a generator.
        Using pytorch dataloaders

        or as a quick hack gets an iterator, and then
        implement pipe in chunks
        """
        _disable = not self._progress_bar
        results = []
        chunks = partition_all(self._chunk, texts)
        total = len(texts) / self._chunk
        for chunk in tqdm(chunks, disable=_disable, total=total):
            for translated in self.pipe(list(chunk)):
                results.append(translated["translation_text"])
        return results

    @staticmethod
    def build_model_name(source, target) -> str:
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
