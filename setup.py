# -*- coding: utf-8 -*-
"""Setup file."""
import numpy as np

from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = cythonize([Extension("metameric.core.metric",
                                  ["metameric/core/metric.pyx"],
                       include_dirs=[np.get_include()])])

setup(name='metameric',
      version='1.0.4',
      description='Interactive activation',
      author='Stéphan Tulkens',
      author_email='stephan.tulkens@uantwerpen.be',
      url='https://github.com/stephantul/metameric',
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
      include_package_data=True)
