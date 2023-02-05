#!/usr/bin/env python3
from pathlib import Path
from setuptools import setup

with open(Path(__file__).parent / 'README.md', 'r') as stream:
    long_description = stream.read()

setup(
    name = 'librosa_loopfinder',
    version = '0.1.0',
    author = 'Kex',
    license='MIT',
    description=('Python library based on librosa for finding seamless music loops.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://github.com/Kexanone/librosa_loopfinder',
    keywords='librosa music music-analysis audio-loops music-loops loop-points',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8'
    ],
    py_modules=['librosa_loopfinder'],
    python_requires=">=3.8",
    install_requires=['librosa>=0.9.2', 'scikit-learn>=1.2.1']
)