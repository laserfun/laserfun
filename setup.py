#!/usr/bin/env python
from setuptools import setup
import os
install_requires=[
      "numpy>=1.15.0",
      "scipy>=1.6.0",
      "sphinx-rtd-theme"
      ]
# on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
#
# if not on_rtd:
#     import numpy as np
#     install_requires=[
#           "numpy>=1.15.0",
#           "scipy>=1.6.0"
#           ]
# else:
#     np = None
#     install_requires=[]

setup(name='laserfun',
      version='0.0.1',
      description='Python functions related to lasers',
      author='laserfun developers',
      author_email='',
      url='https://github.com/laserfun/laserfun',
      install_requires=install_requires,
      packages=['laserfun']
     )
