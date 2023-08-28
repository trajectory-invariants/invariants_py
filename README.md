Python library for the invariant shape descriptors.

[[_TOC_]]

# Getting started

## Installation

Clone (or download) this repository:

    git clone git@gitlab.kuleuven.be:robotgenskill/python_projects/invariants_python.git

Install package:

    cd invariants_python
    python -m pip install -e .


(Optional) To speed-up calculation times, you should additionally install the [fatrop solver](https://gitlab.kuleuven.be/robotgenskill/fatrop/fatrop).

## Examples

Example scripts are found in the `examples` folder.

# Features

The main features are:
- Calculation of invariant descriptors for trajectories.
- Fast trajectory adaptation starting from the invariant descriptors.

Status of current functionality:

| Frenet-Serret Invariants                    | Screw Axis Invariants                             |
| ------------------------------------------- | ------------------------------------------------- |
| {+ global calculation invariants +}         | global calculation invariants                     |
| {+ moving horizon calculation invariants +} | moving horizon calculation invariants             |
| {+ global trajectory generation +}                | global trajectory generation                      |
| {+ moving horizon trajectory generation +}        | moving horizon trajectory generation              |



# History of this repository

- 2018-2019 [Handover thesis of Zeno/Victor](https://gitlab.kuleuven.be/robotgenskill/master_thesis_code/thesis_zenogillis_victorvanwymeersch): The original start of this repository. Functionality was ported from invariants-matlab to Python, mainly focusing on the OCPs for calculating Frenet-Serret invariants and generating trajectories with invariants. Embedded in ROS and tested using data from HTC Vive. The OCPs were implemented in pure Casadi (nlpsol).
- 2019-2021 [etasl invariants integration](https://gitlab.kuleuven.be/robotgenskill/python_projects/etasl_invariants_integration): Extended the trajectory generation code by integrating it with reactive trajectory tracking in eTaSL. The OCPs were reimplemented using optistack. Splines were first used to represent the generated trajectories. 
- 2021-2023: [invariants_python](https://gitlab.kuleuven.be/robotgenskill/python_projects/invariants_python/): Split off everything related to ROS and eTaSL to other repositories, so that this repository is pure Python. Focus was on developing online optimization problems for calculating invariants and trajectory generation. The OCPs were reimplemented in rockit and integrated with fatrop for speeding up the execution.

# Contributors

Main developers: Maxim Vochten, Riccardo Burlizzi

This package has also received contributions by Victor van Wymeersch, Zeno Gillis, Toon Daemen, Glenn Maes, Ali Mousavi, Lander Vanroye



