![install_and_test](https://github.com/trajectory-invariants/invariants_py/actions/workflows/install_and_test.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/trajectory-invariants/invariants_py)
![GitHub issues](https://img.shields.io/github/issues/trajectory-invariants/invariants_py)



Python library for the invariant shape descriptors.



# Getting started

## Installation

### Prerequisites

Prerequisites: Python >3.6 and pip



### Installation of invariants-py

Clone (or download) this repository:

    git clone https://gitlab.kuleuven.be/robotgenskill/public_code/invariants_py.git

Install package in your Python environment:

    cd invariants_py

    pip install --upgrade pip    
    
    pip install -e .

### (Optional) Installation of Fatrop

To speed-up calculation times, you can choose to additionally install the [fatrop solver](https://gitlab.kuleuven.be/robotgenskill/fatrop/fatrop). Currently this solver is only available in Linux.

Clone the Fatrop repository:    

    cd ..

    git clone https://github.com/meco-group/fatrop.git --recursive

    cd fatrop

Set the CMake flags, change the BLASFEO target to your system architecture (see table of https://github.com/giaf/blasfeo)

    sudo apt-get install cmake

    export CMAKE_ARGS="-DBLASFEO_TARGET=X64_AUTOMATIC -DENABLE_MULTITHREADING=OFF"

Build and install the Fatropy project

    cd fatropy
    pip install --upgrade pip setuptools
    pip install .

Install rockit with Fatropy interface

    git clone https://gitlab.kuleuven.be/meco-software/rockit.git
    git clone https://gitlab.kuleuven.be/u0110259/rockit_fatrop_plugin.git ./rockit/rockit/external/fatrop
    cd rockit
    pip install .

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
- 2021-2024: [invariants_py](https://gitlab.kuleuven.be/robotgenskill/python_projects/invariants_py/): Split off everything related to ROS and eTaSL to other repositories, so that this repository is pure Python. Focus was on developing online optimization problems for calculating invariants and trajectory generation. The OCPs were reimplemented in rockit and integrated with fatrop for speeding up the execution.

# Contributors

Main developers: Maxim Vochten, Riccardo Burlizzi, Arno Verduyn

This package has also received contributions by Victor van Wymeersch, Zeno Gillis, Toon Daemen, Glenn Maes, Ali Mousavi, Lander Vanroye



