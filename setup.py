# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 20:25:03 2022

@author: Harley Hanes
"""

from distutils.core import setup

from pathlib import Path
#this_directory = Path(__file__).parent
#long_description = (this_directory / "README.md").read_text()

setup(
    name='UQLibrary',
    version='0.1.2',
    author='H. Hanes',
    author_email='hhanes@ncsu.edu',
    packages=['UQLibrary'],
    url='https://github.com/HarleyHanes/UQLibrary-CourseProject',
    license='LICENSE',
    description='Robust set of sensitivity and identifiability analysis methods.',
    #long_description=long_description,
    install_requires=[
        "numpy >= 1.20.0",
        "scipy >= 1.7.1",
        "matplotlib >= 3.4.3",
        "tabulate >= 0.8.9",
        "mpi4py >= 3.1.3"
    ],
)