# -*- coding: utf-8 -*-
"""Setup file."""
from setuptools import setup
from setuptools import find_packages
from Cython.Build import cythonize
import numpy as np

setup(name='tilapia',
      version='1.0.1',
      description='Interactive activation',
      author='Stéphan Tulkens',
      author_email='stephan.tulkens@uantwerpen.be',
      url='https://github.com/stephantul/tilapia',
      license='MIT',
      packages=find_packages(exclude=['examples']),
      install_requires=['numpy>=1.11.0'],
      classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3'],
      keywords='computational psycholinguistics neural networks',
      zip_safe=False,
      ext_modules=cythonize("tilapia/core/metric.pyx"),
      include_dirs=[np.get_include()]
      )
