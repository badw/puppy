
"""
phonon-unfolding-projections (pup)
"""

from os.path import abspath, dirname
from setuptools import find_packages, setup

setup(
    name='puppy',
    version='1.0.0',
    description='unfolding phonons and plotting with projections',
    url="https://github.com/badw/puppy",
    author="Benjamin A. D. Williamson",
    author_email="benjamin.williamson@ntnu.no",
    license='MIT',
    packages=find_packages(),
    install_requires=['pymatgen','numpy','phonopy','matplotlib','ase']
    )
