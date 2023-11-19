from pathlib import Path
from setuptools import find_packages, setup
import os

def read_requirements(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if not line.isspace()]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pyaging',
    version='v0.0.1',
    python_requires=">=3.8",
    url='https://github.com/rsinghlab/pyaging.git',
    download_url='https://github.com/rsinghlab/pyaging.git',
    author='Lucas Paulo de Lima Camillo',
    author_email='lucas_camillo@alumni.brown.edu',
    description='A Python package for predicting age using various biological clocks with PyTorch.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "docs": read_requirements(os.path.join("docs", "requirements.txt")),
    },
    license="BSD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
