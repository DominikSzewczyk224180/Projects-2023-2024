import sys, os
root_dir = os.path.abspath(__file__)[:(os.path.abspath(__file__).find('ATHENA')+7)]
sys.path.append(root_dir)

from setuptools import setup, find_packages

# Package setup Author: Benjamin Graziadei

setup(
    name="ATHENA",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'athena = scr.pipeline:main'
        ]
    },
)
