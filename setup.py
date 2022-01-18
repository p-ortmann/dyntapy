#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#


from setuptools import setup, find_packages
import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = "See docs on " \
         "[Gitlab](https://gitlab.kuleuven.be/ITSCreaLab/public-toolboxes/dyntapy)"

# PyPI classifiers here
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
]

DESC = (
    "Macroscopic Static and Dynamic Traffic Assignment in Python "
)

INSTALL_REQUIRES = [
    'numba',
    'bokeh',
    'osmnx',
    'pandas',
    'scipy',
    'numpy',
    'geojson',
    'pyyaml',
    'networkx',
    'pyproj',
    'geopandas',
    'Shapely',
    'matplotlib',
    'notebook',
    'setuptools',
]
# now call setup
setup(
    name="dyntapy",
    version="0.2.0",
    description=DESC,
    classifiers=CLASSIFIERS,
    url="https://gitlab.kuleuven.be/ITSCreaLab/mobilitytools",
    author="Paul Ortmann",
    author_email="itscrealab@kuleuven.be",
    long_description=README,
    long_description_content_type="text/markdown",
    license="GPLv3",
    platforms="any",
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(exclude=('testing',)),
    include_package_data=True)
