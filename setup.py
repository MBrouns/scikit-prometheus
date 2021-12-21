import os
from setuptools import setup, find_packages


base_packages = [
    "prometheus-client>=0.12.0",
    "scikit-learn>=1.0.1",
]

docs_packages = [
    "sphinx==1.8.5",
    "sphinx_rtd_theme>=0.4.3",
]
test_packages = [  # we need extras packages for their tests
    "flake8>=3.6.0",
    "pytest>=6.2.5",
    "black>=19.3b0",
    "pytest-cov>=2.6.1",
    "pytest-mock>=1.6.3",
    "pre-commit>=1.18.3",
    "pandas>=1.3.5",
]

dev_packages = docs_packages + test_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="scikit-prometheus",
    version=0.1,
    description="Make it easy to add prometheus metrics to your sklearn models",
    author="Matthijs Brouns",
    packages=find_packages("src"),
    package_dir={"": "src"},
    long_description=read("readme.md"),
    long_description_content_type="text/markdown",
    install_requires=base_packages,
    extras_require={
        "dev": dev_packages,
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
