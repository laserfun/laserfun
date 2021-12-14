#!/usr/bin/env python
from setuptools import setup
import os

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:
    import numpy as np
    install_requires=[
          "numpy>=1.9.2",
          "scipy>=0.15.1"
          ]
else:
    np = None
    install_requires=[]

setup(name='pynlse',
      version='0.0.1',
      description='Nonliner Schrodinger equation in Python',
      author='PyNLSE developers',
      author_email='ycasg@colorado.edu',
      url='https://github.com/pyNLO/PyNLO',
      install_requires=install_requires,
      packages=['nlse']
     )
