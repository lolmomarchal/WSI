# sets up necessary packages

from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="your_package_name",
    version="1.0",
    packages=find_packages(),
    install_requires=requirements,
)
