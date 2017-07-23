#!/usr/bin/env python3
# coding: utf-8

import io

from setuptools import setup, find_packages

# http://blog.ionelmc.ro/2014/05/25/python-packaging/
setup(
  name="tfgraph",
  version="0.2",
  description="Python's Tensorflow Graph Library",
  author="garciparedes",
  author_email="sergio@garciparedes.me",
  url="http://tfgraph.readthedocs.io/en/latest/",
  download_url="https://github.com/tfgraph/tfgraph",
  keywords=[
    "tfg", "bigdata", "tensorflow",
    "graph theory", "pagerank", "university of valladolid",
  ],
  python_requires=">=3.5",
  install_requires=[
    "numpy>=1.11",
    "pandas>=0.20",
    "tensorflow>=1.0",
  ],
  tests_require=[
    "pytest"
  ],
  packages=find_packages(),
  classifiers=[
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
  ],
  long_description=io.open('README.rst', encoding='utf-8').read(),
  include_package_data=True,
  zip_safe=False,
)
