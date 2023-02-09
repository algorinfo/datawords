DeepNLP
==============

Classes and Functions in this package use pytorch, tensor and manly transformers from `Hugging Face <https://huggingface.co>`_.

To use this packages you should install it as:

.. code-block:: bash

                pip install datawords[transformers]



Translator
^^^^^^^^^^^

**Translator** can translate texts from ROMANCE to ENGLISH and viceversa. 

It's based on the `marian models <https://huggingface.co/docs/transformers/main/model_doc/marian>`_

.. autoclass:: datawords.deepnlp.translators.Translator
               :members:


.. autofunction:: datawords.deepnlp.translators.transform_mp
