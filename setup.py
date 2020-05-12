import sys
import os
from setuptools import steup, find_packages
from glob import glob


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="MSTools",
    version="1.0",
    license="MIT License (Please refer LICENSE)",
    description="Tools for analyzing data",
    author="Akagawa Daisuke",
    url="https://github.com/Akasan",
    packages=find_packages("MSTools"),
    package_dir={"": "MSTools"},
    py_modules=[os.splitext(os.basename(path))[0] for path in glob("MSTools/*.py")],
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file("requirements.txt"),
)