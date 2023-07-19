Introduction
==============


You can see this library by modules:

- Parsers: It has tools related to tokenizing and parsing texts.
- Models: It allows to train and load NLP models.
- Indexes: Index tools like sqlite and annoy
- Ranking: PageRank using a more efficient graph library than NetworkX, based on scikit.
- deepnlp: Based on transformers from HuggingFace (optional)

How to start?
--------------

.. code:: bash

          pip install datawords


To use transformers:


.. code:: bash

          pip install datawords[transformers]



How to use?
------------

.. code:: python

      from datawords import parsers

      # loads stop words
      stopw = parsers.load_stop2(lang="en")

      t = """Goodbye world, Hi Fernández. http://chuchu.me/spotify  방 #EEER 😋.\n
        This is the 99th case for 99 days"""
      parsed = parsers.doc_parser(
        t,
        stopw,
        emo_codes=False,
        strip_accents=True,
        numbers=False,
        parse_urls=False
      )
      # Output:
      #  ['goodbye', 'world', 'fernandez', 'bang', '#eeer', 'th', 'case', 'days']
      

Also there is a :class:`datawords.parsers.SentencesParser` which allows to export the configuration as :class:`datawords.parsers.ParserConf`. Besides that, the `SentencesParse` implements :class:`datawords.parsers.ParserProto` protocol.

The final idea of this, is allowing reproductibility between projects or in the same project. 


Project Status
---------------

It's in constant development and not stable yet. Despite that, it's being in use for `Algorinfo <https://algorinfo.com>`_, but be aware that the library could has breaking changes between releases until we can reach the first stable version **1.0** and then follow semantic version.  
          
