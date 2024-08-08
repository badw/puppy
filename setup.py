
"""
phonon-unfolding-projections (puppy)
"""

from os.path import abspath, dirname
from setuptools import find_packages, setup

setup(
    name='puppy',
    version='1.1.0',
    description='unfolding phonons and plotting with projections',
    url="https://github.com/badw/puppy",
    author="Benjamin A. D. Williamson",
    author_email="benjamin.williamson@ntnu.no",
    license='MIT',
    packages=['puppy'],
    install_requires=['tqdm',
                      'pymatgen==2024.4.13',
                      'numpy==1.26.4',
                      'phonopy==2.21.0',
                      'matplotlib',
                      'ase==3.23.0',
                      'doped'],
    python_requires=">=3.10",
    classifiers=[
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Development Status :: 3 - Beta",
            "Intended Audience :: Science/Research",
            "Intended Audience :: System Administrators",
            "Intended Audience :: Information Technology",
            "Operating System :: OS Independent",
            "Topic :: Other/Nonlisted Topic",
            "Topic :: Scientific/Engineering",
        ],
    )
