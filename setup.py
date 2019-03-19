# -*- coding: utf-8 -*-
"""Setup file."""
import numpy as np

from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import re

VERSIONFILE = "metameric/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
mo = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", verstrline, re.M)

if mo:
    version_string = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


extensions = cythonize([Extension("metameric.core.metric",
                                  ["metameric/core/metric.pyx"],
                       include_dirs=[np.get_include()])])

setup(name='metameric',
      version=version_string,
      description='Interactive activation',
      author='Stéphan Tulkens',
      author_email='stephan.tulkens@uantwerpen.be',
      url='https://github.com/stephantul/metameric',
      license='MIT',
      packages=find_packages(exclude=['experiments']),
      install_requires=['numpy>=1.11.0'],
      classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3'],
      keywords='computational psycholinguistics neural networks',
      zip_safe=False,
      ext_modules=extensions,
      include_package_data=True)
