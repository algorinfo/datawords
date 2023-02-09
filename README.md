# datawords

[![PyPI - Version](https://img.shields.io/pypi/v/datawords.svg)](https://pypi.org/project/datawords)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/datawords.svg)](https://pypi.org/project/datawords)
[![readthedocs](https://readthedocs.org/projects/datawords/badge/?version=latest)](https://datawords.readthedocs.io/en/latest/)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install datawords
```

To use transformes from [HuggingFace](https://huggingface.co/) please do:

```console
pip install datawords[transformers]
```

## Quickstart

**deepnlp**:

```python
from datawords.deepnlp import translators
mn = translators.build_model_name("es", "en")
rsp = transform_mp("es", "en", model_path=fp, texts=["hola mundo", "adios mundo", "notias eran las de antes", "Messi es un dios para muchas personas"])

```

## License

`datawords` is distributed under the terms of the [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/) license.
