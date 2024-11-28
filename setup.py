# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

BASEDIR = os.path.abspath(os.path.dirname(__file__))

# 读取requirements.txt文件中的依赖
with open(os.path.join(BASEDIR, "requirements.txt")) as f:
    requirements = f.read().splitlines()

setup(
    name="graphrag_api",
    version="0.3.6",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    author="nightzjp",
    author_email="seven.nighter@gmail.com",
    description="graphrag api 调用",
    long_description=open(os.path.join(BASEDIR, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nightzjp/graphrag_api",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
