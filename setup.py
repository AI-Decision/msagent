from os import path
from setuptools import setup, find_packages

DIR2 = path.dirname(path.abspath(__file__))

setup(

    name='msagent',
    description='A training lib for distributed reinforcement learning based on mirco-service ',
    version='0.0.1',
    author="Junge Zhang, Bailin Wang, Kaiqi Huang",
    packages=find_packages(exclude='Test'),
    include_package_data=True,
    zip_safe = True,
    python_requires=">=3.6",
    install_requires=[
        "pyzmq",
        "python-consul",
        "redis",
        "torch>=1.6",
        "nashpy==0.0.21",
    ]
)