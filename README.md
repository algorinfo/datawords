# datawords

[![PyPI - Version](https://img.shields.io/pypi/v/datawords.svg)](https://pypi.org/project/datawords)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/datawords.svg)](https://pypi.org/project/datawords)
[![readthedocs](https://readthedocs.org/projects/datawords/badge/?version=latest)](https://datawords.readthedocs.io/en/latest/)

-----

This is a library oriented to common and uncommon NLP tasks.


**Datawords** emerge after two years of solving different projects that required NLP techniques 
like training and saving Word2Vec ([Gensim](https://radimrehurek.com/gensim/)) models, finding entities on text ([Spacy](https://spacy.io/)), ranking texts ([scikit-network](https://scikit-network.readthedocs.io/en/latest/)), indexing it ([Spotify Annoy](https://github.com/spotify/annoy)), translating it ([Hugging Face](https://huggingface.co/docs/transformers/index)). 

Then to use those libraries some pre-processing, post-processing tasks and transformations were also required. For this reasons, **datawords exists**. 

Sometimes it’s very opinated (Indexing happens over text, and not over vectors besides Annoy allows it), and sometimes gives you freedom and provide you with helper classes and functions to use freely.

Another way to see this library is as an agreggator of all that excellent libraries mentioned before.

In a nutshell, **Datawords let’s you**:

- Train Word2Vec models (Gensim)
- Build Indexes for texts (Annoy, SQLite)
- Translate texts (Transformers)
- Rank texts (PageRank)


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
