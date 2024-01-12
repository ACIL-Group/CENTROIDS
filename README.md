# CENTROIDS

Code and data for the paper "Enhancing Dimension-Reduced Scatter Plots with Class and Feature Centroids."

| **Zenodo DOI** | **Test Status** | **Coverage** |
|:--------------:|:---------------:|:------------:|
| [![DOI][zenodo-img]][zenodo-url] | [![Build Status][ci-img]][ci-url] | [![Codecov][codecov-img]][codecov-url] |

[zenodo-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.10465863.svg
[zenodo-url]: https://doi.org/10.5281/zenodo.10465863

[ci-img]: https://github.com/ACIL-Group/CENTROIDS/actions/workflows/Test.yml/badge.svg
[ci-url]: https://github.com/ACIL-Group/CENTROIDS/actions/workflows/Test.yml

[codecov-img]: https://codecov.io/gh/ACIL-Group/CENTROIDS/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/ACIL-Group/CENTROIDS

## Table of Contents

- [CENTROIDS](#centroids)
  - [Table of Contents](#table-of-contents)
  - [Summary](#summary)
  - [Usage](#usage)
    - [Setup](#setup)
    - [Execution](#execution)

## Summary

This project implements the code and data necessary to generate the figures and results of the paper "Enhancing Dimension-Reduced Scatter Plots with Class and Feature Centroids."
This repository contains multiple python scripts for generating the various results of the paper with multiple `data -> script -> output` workflows.

The project is laid out as follows:

- `src/`: the Python files implementing each experiment.
- `data/`: the data files necessary to run the experiment.
- `example_output/`: example files demonstrating the output of the experiment.

## Usage

Brief installation instructions can be found in [Setup](#setup), and use of the scripts can be found in [Execution](#execution).

### Setup

Create and activate a virtual Python environment with your favorite tool (e.g., `conda`, `mamba`, or `venv`).

For example, with `conda`:

```shell
conda create -n centroids python=3.11
conda activate centroids
```

Next, install dependencies while inside this virtual environment via the `requirements.txt` file at the top of this repo:

```shell
pip install -r requirements.txt
```

### Execution

Each experiment lives in Python files under the `src/` directory.
To execute each experiment, simply run the file through the Python interpreter while in your virtual environment as follows:

```shell
python src/<experiment>.py
```

where `<experiment>` is the name of the file that you wish to run.

Examples of the output files that are generated during these experiments can be seen in the `example_output/` directory.
