"""
Minimal setup.py for backward compatibility with tools that do not yet
consume pyproject.toml directly.  All authoritative project metadata lives
in pyproject.toml; this file simply delegates to setuptools.
"""
from setuptools import setup, find_packages

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
