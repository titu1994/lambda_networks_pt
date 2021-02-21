from setuptools import setup, find_packages
import re
import os
import io


# Readme longform doc
this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# Requirements
with open('requirements.txt', 'r') as f:
    requirements = f.readlines()
    requirements = [line.replace("\n", "").strip() for line in requirements]


def get_version(package):
    """Return package version as listed in `__version__` in `init.py`."""
    init_py = open(os.path.join(package, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


setup(
    name='lambda_networks_pt',
    version=get_version("lambda_networks"),
    packages=find_packages(exclude=['tests']),
    url='https://github.com/titu1994/lambda_networks',
    license='MIT',
    author='Somshubra Majumdar',
    author_email='titu1994@gmail.com',
    description='Pytorch implementation of Lambda Networks (https://arxiv.org/abs/2102.08602)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    extras_require={
        'tests': ['pytest'],
    },
    classifiers=(
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ),
    zip_safe=False,
    test_suite="tests",
)
