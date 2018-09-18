# -*- coding: utf-8 -*-
"""Setup file."""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(name='tilapia',
      version='1.0.2',
      description='Interactive activation',
      author='Stéphan Tulkens',
      author_email='stephan.tulkens@uantwerpen.be',
      url='https://github.com/stephantul/tilapia',
      license='MIT',
      install_requires=['numpy>=1.11.0', 'cython'],
      classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3'],
      keywords='computational psycholinguistics neural networks',
      zip_safe=False,
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("tilapia.core.metric",
                             ["tilapia/core/metric.pyx"])],
      include_dirs=[np.get_include()],
      include_package_data=True
      )
