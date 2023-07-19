.. datawords documentation master file, created by
   sphinx-quickstart on Tue Feb  7 10:33:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to datawords!
=====================================

This is a library oriented to common and uncommon NLP tasks.

**Datawords** emerges after two years of solving different projects that required NLP techniques like training and saving Word2Vec (`Gensim <https://radimrehurek.com/gensim/>`_) models, finding entities on text (`Spacy <https://spacy.io/>`_ ), ranking texts (`scikit-network <https://scikit-network.readthedocs.io/en/latest/>`_), indexing it (`Spotify Annoy <https://github.com/spotify/annoy>`_), translating it (`Hugging Face <https://huggingface.co/docs/transformers/index>`_).

Then to use that libraries some pre-processing, post-processing tasks and transformations were also required.
For this reason, **datawords** exists. Sometimes it's very opinated (Indexing is over text, and not over vectors like Annoy allows.), sometimes gives you freedom and abstract classes to expand the functionality.

Another way to see this library, it's as an agreggator of all that excellent libraries mentioned before. 

In a nutshell, **Datawords** let's you:

* Train Word2Vec models (Gensim)
* Build Indexes for texts (Annoy, SQLite)
* Translate texts (Transformers)
* Rank texts (PageRank)


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   api_reference



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
