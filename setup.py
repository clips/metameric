# -*- coding: utf-8 -*-
"""Setup file."""
import sys
from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension
import numpy as np


cython = False

if "--cython" in sys.argv:
    cython = True
    sys.argv.remove("--cython")

if cython:
    from Cython.Build import cythonize
    extensions = cythonize([Extension("tilapia.core.metric",
                                      ["tilapia/core/metric.pyx"])])
else:
    extensions = [Extension("tilapia.core.metric",
                            ["tilapia/core/metric.c"])]

setup(name='tilapia',
      version='1.0.3',
      description='Interactive activation',
      author='Stéphan Tulkens',
      author_email='stephan.tulkens@uantwerpen.be',
      url='https://github.com/stephantul/tilapia',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy>=1.11.0'],
      classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3'],
      keywords='computational psycholinguistics neural networks',
      zip_safe=False,
      ext_modules=extensions,
      include_dirs=[np.get_include()],
      include_package_data=True)
