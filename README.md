![install_and_test](https://github.com/trajectory-invariants/invariants_py/actions/workflows/install_and_test.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/trajectory-invariants/invariants_py)
![GitHub issues](https://img.shields.io/github/issues/trajectory-invariants/invariants_py)

`invariants-py` is a Python library to robustly calculate coordinate-invariant trajectory representations using geometric optimal control. 
It also supports trajectory generation under user-specified trajectory constraints starting from invariant representations.

More information can be found on the documentation website: https://trajectory-invariants.github.io

## Features

<!-- TODO: Screenshots or gifs to show results. -->

The main features are:
- Calculation of invariant descriptors for trajectories.
- Fast trajectory adaptation starting from the invariant descriptors.

Invariant trajectory representations find their application in trajectory analysis, trajectory segmentation, recognition and generalization. 

## Installation

The package can be installed from PyPI or from source.

### Prerequisites

The package requires Python 3.8 or higher.

### 1. Installation from PyPI

This installation option is recommended if you only want to use the package and do not plan to modify the source code.

Upgrade your version of `pip`:
```shell
pip install --upgrade pip    
```

Install the package:
```shell
pip install invariants-py
```

### 2. Installation from source

This installation option is recommended if you plan to modify or experiment with the source code.

Clone (or download) the `invariants-py` repository:
```shell
git clone https://github.com/trajectory-invariants/invariants_py.git
```

Navigate to the cloned repository:
```shell
cd invariants_py
```

Upgrade your version of `pip`:
```shell
pip install --upgrade pip    
```

Install the package:
```shell
pip install -e .
```

## Getting started

<!-- Add a code snippet here with basic example in a few lines -->

Basic examples are provided to get you started in the [examples folder](./examples/).

More detailed examples and tutorials can be found on the [documentation website](https://trajectory-invariants.github.io/docs/python/).

## Speed up using Fatrop

To speed up the solution of the optimal control problems, you can optionally install the [fatrop solver](https://gitlab.kuleuven.be/robotgenskill/fatrop/fatrop). The instructions are available on [this page](https://trajectory-invariants.github.io/docs/python/installation/installation-fatrop/) of the documentation website.

## Roadmap

The following features are planned for future releases:
- Support for more types of invariant representations (e.g. screw invariants, global invariants, ...).
- Support for more types of constraints in the trajectory generation.
- Benchmarking between different invariant representations in terms of robustness, computational efficiency, and generalizability.

## Contributing

We welcome contributions to this repository, for example in the form of pull requests.

## Contributors

We wish to thank the following people for their contributions to an early version of the software: Victor van Wymeersch, Zeno Gillis, Toon Daemen, Glenn Maes, Ali Mousavi, Lander Vanroye

## Support

For questions, bugs, feature requests, etc., please open [an issue on this repository](https://github.com/trajectory-invariants/invariants_py/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citing

If you use this package in your research, please cite the following paper:

```
@article{vochten2023invariant,
  title={Invariant Descriptors of Motion and Force Trajectories for Interpreting Object Manipulation Tasks in Contact},
  author={Vochten, Maxim and Mohammadi, Ali Mousavi and Verduyn, Arno and De Laet, Tinne and Aertbeli{\"e}n, Erwin and De Schutter, Joris},
  journal={IEEE Transactions on Robotics},
  year={2023},
  volume={39},
  number={6},
  pages={4892-4912},
  doi={10.1109/TRO.2023.3309230}}
```