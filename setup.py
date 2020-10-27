#!/usr/bin/env python3

from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = []
with open(path.join(here, "requirements.txt"), "r") as infile:
    requirements = [line for line in infile.read().split("\n") if line]

setup(
    name="anim_utils",
    version="0.1",
    description="Skeleton Animation Utilities.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eherr/anim_utils",
    author="DFKI GmbH",
    license='MIT',
    keywords="skeleton animation data retargeting",
    packages=find_packages(exclude=("examples",)),
    python_requires=">=3.5.*, <4",
    install_requires=requirements,
)
