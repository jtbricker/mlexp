from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name="mlexp",
    version="0.0.1",
    author="Justin Bricker",
    author_email="jt.bricker@gmail.com",
    description="A package with utilities to help in machine learning analyses",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/jtbricker/mlexp/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
