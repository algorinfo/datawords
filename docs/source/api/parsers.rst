Parsers
==============

Parsers module contains mostly functions and classes to parse texts. 

doc_parser
^^^^^^^^^^^

**doc_parser** is the most critical function in the package. It's used
internally by multiple classes. 

This function get a text and produce the tokens needed to feed diferent models.


.. autofunction:: datawords.parsers.doc_parser


.. autoclass:: datawords.parsers.ParserConf
               :members:


Sentences Parser
^^^^^^^^^^^^^^^^
                 
.. autoclass:: datawords.parsers.ParserProto
               :members:

.. autoclass:: datawords.parsers.SentencesParser
               :members:


.. autoclass:: datawords.parsers.SentencesIterator
               :members:



PhrasesModel
^^^^^^^^^^^^^^^^^^

The **PhrasesModel** is used to build bgrams.
In this case, it's a wrapper around `Gensim Phrases <https://radimrehurek.com/gensim/models/phrases.html>`_

.. autoclass:: datawords.parsers.PhrasesModel
               :members:



                  
Other functions
^^^^^^^^^^^^^^^^

.. autofunction:: datawords.parsers.parser_from_conf

.. autofunction:: datawords.parsers.load_stop

.. autofunction:: datawords.parsers.generate_ngrams

.. autofunction:: datawords.parsers.norm_token

.. autofunction:: datawords.parsers.apply_regex

