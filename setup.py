# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('LICENSE') as f:
  license = f.read()

setup(
  name='sound-classification',
  version='1.0.0',
  description='Some sound classification fun',
  author='Joe D\'Alessandro',
  author_email='joe@thejoeshow.net',
  url='https://github.com/xupit3r/sound-classification',
  license=license,
  packages=find_packages()
)

