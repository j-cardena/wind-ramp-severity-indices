"""
Setup script for Wind Power Ramp Severity Indices package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ramp-severity-indices",
    version="1.0.0",
    author="Julian Cardenas-Barrera",
    author_email="julian.cardenas@unb.ca",
    description="Novel shape-based severity indices for wind power ramp events",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/wind-ramp-severity-indices",
    project_urls={
        "Bug Tracker": "https://github.com/[username]/wind-ramp-severity-indices/issues",
        "Documentation": "https://github.com/[username]/wind-ramp-severity-indices#readme",
        "Paper": "https://doi.org/10.xxxx/xxxxx",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
)
