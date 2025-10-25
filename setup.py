"""
Setup script for llava_headct package.
This allows the package to be installed in development mode using:
pip install -e .

This makes imports from src work reliably from anywhere.
"""

from setuptools import setup, find_packages

setup(
    name="llava_headct",
    version="0.1.0",
    description="LLaVA Head CT Analysis Package",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "torchvision",
        "polars",
        "numpy",
        "pandas",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ]
    },
)
