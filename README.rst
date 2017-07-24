TFGraph: Python's Tensorflow Graph Library
=======

.. |travisci| image:: https://img.shields.io/travis/tfgraph/tfgraph/master.svg?style=flat-square
   :target: https://travis-ci.org/tfgraph/tfgraph

.. |codecov| image:: https://img.shields.io/codecov/c/github/tfgraph/tfgraph.svg?style=flat-square
   :target: https://codecov.io/gh/tfgraph/tfgraph?branch=master

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square
   :target: http://tfgraph.readthedocs.io/en/latest/?badge=latest

.. |gitter| image:: https://img.shields.io/gitter/tfgraph/tfgraph.svg?style=flat-square
   :target: https://gitter.im/tfgraph/tfgraph?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. |license| image:: https://img.shields.io/github/license/tfgraph/tfgraph.svg?style=flat-square
    :target: https://github.com/tfgraph/tfgraph

.. |release| image:: https://img.shields.io/github/release/tfgraph/tfgraph.svg?style=flat-square
    :target: https://github.com/tfgraph/tfgraph

|travisci| |codecov| |docs| |gitter| |license| |release|

Description
-----------
This work consists of a study of a set of techniques and strategies related with algorithm's design, whose purpose is the resolution of problems on massive data sets, in an efficient way. This field is known as Algorithms for Big Data. In particular, this work has studied the Streaming Algorithms, which represents the basis of the data structures of sublinear order o(n) in space, known as Sketches. In addition, it has deepened in the study of problems applied to Graphs on the Semi-Streaming model. Next, the PageRank algorithm was analyzed as a concrete case study. Finally, the development of a library for the resolution of graph problems, implemented on the top of the intensive mathematical computation platform known as TensorFlow has been started.

Content
-------
* `Source Code <https://github.com/tfgraph/tfgraph/blob/master/src/tfgraph>`__
* `API Documentation <http://tf-g.readthedocs.io/>`__
* `Code Examples <https://github.com/tfgraph/tfgraph/blob/master/examples>`__
* `Tests <https://github.com/tfgraph/tfgraph/blob/master/tests>`__


How to install
--------------

If you have git installed, you can try::

    $ pip install git+https://github.com/tfgraph/tfgraph.git

If you get any installation or compilation errors, make sure you have the latest pip and setuptools::

    $ pip install --upgrade pip setuptools

How to run the tests
--------------------

Install in editable mode and call `pytest`::

    $ pip install -e .
    $ pytest
