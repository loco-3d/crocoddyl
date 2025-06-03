
<p align="center">
  <img src="./doc/images/crocoddyl_logo.png" width="800" alt="Crocoddyl Logo" align="center"/>
</p>

<p align="center">
<a href="https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/master/doxygen-html/"><img src="https://img.shields.io/badge/docs-online-brightgreen" alt="Documentation"/></a>
<a href="https://gepgitlab.laas.fr/loco-3d/crocoddyl/pipelines?ref=devel"><img src="https://gepgitlab.laas.fr/loco-3d/crocoddyl/badges/devel/pipeline.svg">
<a href="https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/devel/coverage/"><img src="https://gepgitlab.laas.fr/loco-3d/crocoddyl/badges/devel/coverage.svg">
<a href="https://repology.org/project/crocoddyl/versions"><img src="https://repology.org/badge/version-for-repo/aur/crocoddyl.svg" alt="AUR package"></a>
<a href="https://anaconda.org/conda-forge/crocoddyl"><img src="https://img.shields.io/conda/vn/conda-forge/crocoddyl.svg">
<a href="https://repology.org/project/crocoddyl/versions"><img src="https://repology.org/badge/version-for-repo/nix_stable_24_05/crocoddyl.svg" alt="nixpkgs stable 24.05 package"></a>
<a href="https://repology.org/project/crocoddyl/versions"><img src="https://repology.org/badge/version-for-repo/nix_unstable/crocoddyl.svg" alt="nixpkgs unstable package"></a>
<a href="https://pypi.org/project/crocoddyl/"><img src="https://badge.fury.io/py/crocoddyl.svg">
<a href="https://anaconda.org/conda-forge/crocoddyl"><img src="https://anaconda.org/conda-forge/crocoddyl/badges/downloads.svg">
<a href="https://tldrlegal.com/license/bsd-3-clause-license-%28revised%29#fulltext"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg?style=flat">
<!-- <a href="https://github.com/loco-3d/crocoddyl/graphs/contributors"><img src="https://img.shields.io/github/contributors/loco-3d/crocoddyl"> -->
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" /></a>
</p>

## <img align="center" height="20" src="https://i.imgur.com/vAYeCzC.png"/> Introduction

<img align="right" src="https://i.imgur.com/o2LfbDq.gif" width="25%"/>

**[Crocoddyl](https://cmastalli.github.io/publications/crocoddyl20icra.html)** is an optimal control library for robot control under contact sequence. Its solvers are based on novel and efficient differential dynamic programming (DDP) algorithms. **Crocoddyl** computes optimal trajectories and feedback gains. It uses **[Pinocchio](https://github.com/stack-of-tasks/pinocchio)** for fast computation of robot dynamics and analytical derivatives.

If you want to learn more about **Crocoddyl** and its solvers, we suggest reading [[1]](#1) [[2]](#2) [[3]](#3) and visiting [PUBLICATIONS.md](https://github.com/loco-3d/crocoddyl/blob/master/PUBLICATIONS.md). If you want to follow the current developments and contribute, please directly refer to the [devel branch](https://github.com/loco-3d/cddp/tree/devel).


## :crocodile: Crocoddyl features
<table >
  <tr>
    <td align="left"><img src="https://cmastalli.github.io/assets/img/publications/highly_dynamic_maneuvers.png" width="10000"/></td>
    <td align="right"><img src="https://i.imgur.com/RQR2Ovx.gif"/> <img src="https://i.imgur.com/kTW0ePh.gif"/></td>
  </tr>
</table>

**Crocoddyl** is versatile:
 * Various optimal control solvers (DDP, FDDP, BoxFDDP, Ipopt, etc)
 * Analytical and sparse derivatives via **[Pinocchio](https://github.com/stack-of-tasks/pinocchio)**
 * Differential geometry support leveraging **[Pinocchio](https://github.com/stack-of-tasks/pinocchio)**
 * Various integrators, dynamics, costs and constraints
 * Numerical differentiation support
 * Automatic differentiation support via **[CppAD](https://github.com/coin-or/CppAD)**

**Crocoddyl** is efficient and flexible:
 * Cache friendly
 * Multi-threading support via **[OpenMP](https://www.openmp.org/)**
 * Python bindings (including abstractions) via **[Boost Python](https://wiki.python.org/moin/boost.python)**
 * C++11/14/17/20 compliant
 * Extensively tested
 * Code generation in both C++ and Python via **[CppADCoGen](https://github.com/joaoleal/CppADCodeGen)**

## :penguin: Installation

**Crocoddyl** can be easily installed on various Linux (Ubuntu, Fedora, etc.) and Unix distributions (Mac OS X, BSD, etc.). Below, there are different ways to install Crocoddyl.

### <img src="https://aur.archlinux.org/static/images/favicon.ico" height="18" /> On ArchLinux

With your favorite AUR helper, eg. paru:
```bash
    paru -Syu crocoddyl
```

### :dragon: From <img src="https://s3.amazonaws.com/conda-dev/conda_logo.svg" height="18">

Just run the following command in the terminal:
```bash
   conda install crocoddyl -c conda-forge
```
Conda installation supports [![conda install](https://anaconda.org/conda-forge/crocoddyl/badges/platforms.svg)](https://anaconda.org/conda-forge/crocoddyl).

### <img src="https://avatars.githubusercontent.com/u/487568" height="18" /> From Nix

`crocoddyl` & `python3Packages.crocoddyl` are available in [nixpkgs](https://github.com/NixOS/nixpkgs/).

This repository is also a flake, so you may:
- run a python shell with crocoddyl: `nix run github:loco-3d/crocoddyl`
- use it in your own flake: `crocoddyl.url = "github:loco-3d/crocoddyl";`

The build cache use by CI and developers is [gepetto.cachix.org](https://app.cachix.org/cache/gepetto)

### :snake: From <img src="https://res.cloudinary.com/practicaldev/image/fetch/s--4-K6Sjm4--/c_imagga_scale,f_auto,fl_progressive,h_420,q_auto,w_1000/https://cdn-images-1.medium.com/max/1600/1%2A_Wkc-WkNu6GJAAQ26jXZOg.png" height="22">

Just run the following command in the terminal:
```bash
  pip install --user crocoddyl
```

### :turtle: With ROS

Just clone it (with `--recursive`) into a catkin workspace and compile it.

### :package: From Debian / Ubuntu packages, with [robotpkg](http://robotpkg.openrobots.org)

1. If you have never added robotpkg's software repository, [do it now](http://robotpkg.openrobots.org/debian.html):
   ```bash
   sudo tee /etc/apt/sources.list.d/robotpkg.list <<EOF
   deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg
   EOF

   curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
   sudo apt update
   ```
2. Install Crocoddyl and its Python bindings:
   ```bash
   sudo apt install robotpkg-py3\*-crocoddyl
   ```
3. Configure your environment variables:
   ```bash
   export PATH=/opt/openrobots/bin:$PATH
   export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
   export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
   export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH
   ```

### :file_folder: From source

1. Install Crocoddyl's mandatory dependencies:
   * [pinocchio](https://github.com/stack-of-tasks/pinocchio)
   * [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
   * [eigenpy](https://github.com/stack-of-tasks/eigenpy)
   * [Boost](https://www.boost.org/)
2. (optional) Install Crocoddyl's optional dependencies
   * [OpenMP](https://www.openmp.org/) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(for multi-threading support)
   * [CppADCoGen](https://github.com/joaoleal/CppADCodeGen) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(for code-generation support)
   * [pycppad](https://github.com/Simple-Robotics/pycppad) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(for code-generation support)
   * [Ipopt](https://github.com/coin-or/Ipopt) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(for Ipopt support)
   * [example-robot-data](https://github.com/gepetto/example-robot-data) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(for robotic examples, install Python loaders)
   * [gepetto-viewer-corba](https://github.com/Gepetto/gepetto-viewer-corba) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(for display in Gepetto viewer, i.e., `GepettoDisplay`)
   * [meshcat-python](https://github.com/rdeits/meshcat-python) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(for display in Meshcat, i.e., `MeshcatDisplay`)
   * [whole_body_state_rviz_plugin](https://github.com/loco-3d/whole_body_state_rviz_plugin) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(for display in ROS, i.e., `RvizDisplay`)
   * [crocoddyl_msgs](https://github.com/RobotMotorIntelligence/crocoddyl_msgs) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(for display in ROS, i.e., `RvizDisplay`)
   * [urdf_parser_py](https://github.com/ros/urdf_parser_py) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(for display in ROS, i.e., `RvizDisplay`)
   * [jupyter](https://jupyter.org/) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(for notebooks)
   * [matplotlib](https://matplotlib.org/) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(for plotting)
3. Clone it (with --recursive), create a build directory inside, and:
   ```bash
   cmake .. && make && make install
   ```
You can disable the tests, examples, and benchmarks using the following CMake options: `BUILD_TESTING`, `BUILD_EXAMPLES`, and `BUILD_BENCHMARK`. Note that the tests, examples or benchmarks require [example-robot-data](https://github.com/gepetto/example-robot-data).

## :mag: Documentation

Crocoddyl's Doxygen documentation is available [here](https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/master/doxygen-html/). Alternatively, you can also check out the Jupyter notebooks. We suggest to explore at least these notebooks:
- [notebooks/01_introduction_to_crocoddyl.ipynb](https://github.com/loco-3d/crocoddyl/blob/devel/notebooks/01_introduction_to_crocoddyl.ipynb)
- [notebooks/02_optimizing_a_cartpole_swingup.ipynb](https://github.com/loco-3d/crocoddyl/blob/devel/notebooks/02_optimizing_a_cartpole_swingup.ipynb)
- [notebooks/03_optimizing_an_unicycle.ipynb](https://github.com/loco-3d/crocoddyl/blob/devel/notebooks/03_optimizing_an_unicycle.ipynb)
- [notebooks/04_actuating_an_acrobot.ipynb](https://github.com/loco-3d/crocoddyl/blob/devel/notebooks/04_actuating_an_acrobot.ipynb)
- [notebooks/05_codegenerating_a_cartpole.ipynb](https://github.com/loco-3d/crocoddyl/blob/devel/notebooks/05_codegenerating_a_cartpole.ipynb)
- [notebooks/06_scaling_to_robotics.ipynb](https://github.com/loco-3d/crocoddyl/blob/devel/notebooks/06_scaling_to_robotics.ipynb)

Moreover, after installation, you could run the examples as follows:
```bash
python -m crocoddyl.examples.quadrupedal_gaits "display" "plot" # enable display and plot
```
or run examples, unit tests and benchmarks from your build directory as
```bash
cd build
make test
make -s examples-quadrupedal_gaits INPUT="display plot" # enable display and plot
make -s benchmarks-cpp-quadrupedal_gaits INPUT="100 walk" # number of trials ; type of gait
```
where it is possible to enable display and/or plots generated by our examples using the environment variables:
```bash
export CROCODDYL_DISPLAY=1
export CROCODDYL_PLOT=1
```

## :telescope: Citing Crocoddyl

To cite **Crocoddyl** in your academic research, please use the following BibTeX lines:
```bibtex
@inproceedings{mastalli20crocoddyl,
  author={Mastalli, Carlos and Budhiraja, Rohan and Merkt, Wolfgang and Saurel, Guilhem and Hammoud, Bilal
  and Naveau, Maximilien and Carpentier, Justin and Righetti, Ludovic and Vijayakumar, Sethu and Mansard, Nicolas},
  title={{Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control}},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year={2020}
}
```
Please consider citing our selected publications and contributions described in [PUBLICATIONS.md](https://github.com/loco-3d/crocoddyl/blob/devel/PUBLICATIONS.md).

**Crocoddyl's** contributions extend beyond efficient software development. Please also consider citing the algorithm contributions of our different solvers and formulations:
 - Feasibility-driven solvers (FDDP and Box-FDDP): [[1]](#1), [[2]](#2)
 - Inverse-dynamics trajectory optimization and endpoint constrains: [[3]](#3) [[4]](#4)
 - Multi-Contact Inertial Parameters Estimation and Localization in Legged Robots: [[5]](#5)


Finally, please also consider citing **[Pinocchio](https://github.com/stack-of-tasks/pinocchio)**, which contributes to the efficient implementation of rigid body algorithms and their derivatives. For more details on how to cite Pinocchio visit: [https://github.com/stack-of-tasks/pinocchio](https://github.com/stack-of-tasks/pinocchio).


### :open_book:  Selected publications

<a id="1">[1]</a> C. Mastalli, R. Budhiraja, W. Merkt, G. Saurel, B. Hammoud, M. Naveau, J. Carpentier, L. Righetti, S. Vijayakumar and N. Mansard. [Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control](https://cmastalli.github.io/publications/crocoddyl20icra.html), IEEE International Conference on Robotics and Automation (ICRA), 2020.

<a id="2">[2]</a> C. Mastalli, W. Merkt, J. Marti-Saumell, H. Ferrolho, J. Sola, N. Mansard and S. Vijayakumar. [A Feasibility-Driven Approach to Control-Limited DDP](https://arxiv.org/pdf/2010.00411.pdf), Autonomous Robots, 2022.

<a id="3">[3]</a> C. Mastalli, S. P. Chhatoi, T. Corbères, S. Tonneau and S. Vijayakumar. [Inverse-Dynamics MPC via Nullspace Resolution](https://arxiv.org/pdf/2209.05375.pdf), IEEE Transactions on Robotics, 2023.

<a id="4">[4]</a> M. Parilli, S. Martinez, and C.Mastall. [Endpoint-Explicit Differential Dynamic Programming via Exact Resolution](https://arxiv.org/pdf/2503.03897), IEEE International Conference on Robotics and Automation (ICRA), 2025.

<a id="5">[5]</a> S. Martinez, R. Griffin, and C.Mastalli. [Multi-Contact Inertial Parameters Estimation and Localization in Legged Robots](https://arxiv.org/pdf/2403.17161), IEEE Robotics and Automation Letters (RAL), 2025.

## :computer:  Questions and Issues

You have a question or an issue? Please open a [new issue](https://github.com/loco-3d/crocoddyl/issues) or a [discussion](https://github.com/loco-3d/crocoddyl/discussions).


## :copyright: Credits

### :writing_hand: Written by

- [Carlos Mastalli](https://cmastalli.github.io/), Heriot-Watt University :uk: (project manager)
- [Nicolas Mansard](http://projects.laas.fr/gepetto/index.php/Members/NicolasMansard), LAAS-CNRS :fr:
- [Rohan Budhiraja](https://scholar.google.com/citations?user=NW9Io9AAAAAJ), LAAS-CNRS :fr: (alumnus)

### :construction_worker: With contributions from


- [Guilhem Saurel](http://projects.laas.fr/gepetto/index.php/Members/GuilhemSaurel), LAAS-CNRS :fr:
- [Wolfgang Merkt](http://www.wolfgangmerkt.com/research/), University of Oxford :uk:
- [Justin Carpentier](https://jcarpent.github.io/), INRIA :fr:
- [Andrea Del Prete](https://andreadelprete.github.io/), Università degli Studi di Trento :it:
- [Wilson Jallet](https://manifoldfr.github.io), INRIA :fr:
- [Maximilien Naveau](https://scholar.google.fr/citations?user=y_-cGlUAAAAJ&hl=fr), MPI :de:
- [Josep Martí Saumell](https://www.iri.upc.edu/staff/jmarti), IRI: CSIC-UPC :es:
- [Bilal Hammoud](https://scholar.google.com/citations?hl=en&user=h_4NKpsAAAAJ), MPI :de:
- [Julian Eßer](https://github.com/julesser), Fraunhofer :de:


## :trophy: Acknowledgments

**Crocoddyl** development was supported by the [EU MEMMO project](http://www.memmo-project.eu/) and the [EU RoboCom++ project](http://robocomplusplus.eu/). It is maintained by the [Robot Motor Intelligence (RoMI) Lab](https://www.romilab.org) [@ Heriot-Watt University](https://www.edinburgh-robotics.org/), the [Gepetto team](http://projects.laas.fr/gepetto/) [@ LAAS-CNRS](http://www.laas.fr), and the [Willow team](https://www.di.ens.fr/willow/) [@ INRIA](https://www.inria.fr/fr/centre-inria-de-paris).
