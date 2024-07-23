import os
from setuptools import setup, find_packages
import unittest

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

setup(
    name="d2pc",
    version="0.0.1",
    author="Haldun Balim",
    author_email="haldunbalim@gmail.com",
    description=("no description yet"),
    license="BSD",
    keywords="no keywords yet",
    packages=find_packages(),
    test_suite='setup.get_test_suite',
    long_description=read('README.MD'),
    classifiers=[

    ],
)
