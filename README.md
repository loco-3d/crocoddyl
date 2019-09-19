Contact RObot COntrol by Differential DYnamic programming Library (crocoddyl)
===============================================

## <img align="center" height="20" src="https://i.imgur.com/vAYeCzC.png"/> Introduction

**Crocoddyl** is an optimal control library for robot control under contact sequence. Its solver is based on an efficient Differential Dynamic Programming (DDP) algorithm. **Crocoddyl** computes optimal trajectories along to optimal feedback gains. It uses **Pinocchio** for fast computation of robot dynamics and its analytical derivatives.

The source code is released under the [BSD 3-Clause license](LICENSE).

**Authors:** [Carlos Mastalli](https://cmastalli.github.io/) and Rohan Budhiraja <br />
**Instructors:** Justin Carpentier and Nicolas Mansard <br />
**With additional support from the Gepetto team at LAAS-CNRS.**

[![License BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg?style=flat)](https://tldrlegal.com/license/bsd-3-clause-license-%28revised%29#fulltext)
[![pipeline status](https://gepgitlab.laas.fr/loco-3d/crocoddyl/badges/devel/pipeline.svg)](https://gepgitlab.laas.fr/loco-3d/crocoddyl/pipelines?ref=devel)
[![coverage report](https://gepgitlab.laas.fr/loco-3d/crocoddyl/badges/devel/coverage.svg)](https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/devel/coverage/)
[![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/loco-3d/crocoddyl)](https://img.shields.io/github/v/tag/loco-3d/crocoddyl)
[![GitHub repo size](https://img.shields.io/github/repo-size/loco-3d/crocoddyl)](https://img.shields.io/github/repo-size/loco-3d/crocoddyl)
[![contributors](https://img.shields.io/github/contributors/loco-3d/crocoddyl)](https://img.shields.io/github/contributors/loco-3d/crocoddyl)

[![GitHub Release Date](https://img.shields.io/github/release-date/loco-3d/crocoddyl)](https://img.shields.io/github/release-date/loco-3d/crocoddyl)
[![GitHub last commit](https://img.shields.io/github/last-commit/loco-3d/crocoddyl)](https://img.shields.io/github/last-commit/loco-3d/crocoddyl)

If you want to follow the current developments, you can directly refer to the [devel branch](https://gepgitlab.laas.fr/loco-3d/cddp/tree/devel).


## <img align="center" height="20" src="https://i.imgur.com/x1morBF.png"/> Installation
**Crocoddyl** can be easily installed on various Linux (Ubuntu, Fedora, etc.) and Unix distributions (Mac OS X, BSD, etc.). Please refer to 

### Installation through robotpkg

You can install this package throught robotpkg. robotpkg is a package manager tailored for robotics softwares. It greatly simplifies the release of new versions along with the management of their dependencies. You just need to add the robotpkg apt repository to your sources.list and then use `sudo apt install robotpkg-py27-crocoddyl`:

If you have never added robotpkg as a softwares repository, please follow first the instructions from 1 to 3. Otherwise, go directly to instruction 4. Those instructions are similar to the installation procedures presented in [http://robotpkg.openrobots.org/debian.html](http://robotpkg.openrobots.org/debian.html).

1. Add robotpkg as source repository to apt:

		sudo tee /etc/apt/sources.list.d/robotpkg.list <<EOF
		deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -sc) robotpkg
		deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg
		EOF

2. Register the authentication certificate of robotpkg:

		curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -

3. You need to run at least once apt update to fetch the package descriptions:

		sudo apt-get update

4. The installation of Crocoddyl:

		sudo apt install robotpkg-py27-crocoddyl # for Python 2

		sudo apt install robotpkg-py35-crocoddyl # for Python 3

Finally you will need to configure your environment variables, e.g.:

		export PATH=/opt/openrobots/bin:$PATH
		export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
		export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
		export PYTHONPATH=/opt/openrobots/lib/python2.7/site-packages:$PYTHONPATH


### Building from source

**Crocoddyl** is c++ library with Python bindings for versatible and fast prototyping. It has the following dependecies:

* [pinocchio](https://github.com/stack-of-tasks/pinocchio)
* [quadprog](https://pypi.org/project/quadprog/)
* [multicontact-api](https://gepgitlab.laas.fr/loco-3d/multicontact-api)
* [example-robot-data](https://gepgitlab.laas.fr/gepetto/example-robot-data) (optional for running examples, install Python loaders)
* [gepetto-viewer-corba](https://github.com/Gepetto/gepetto-viewer-corba) (optional for running examples and notebooks)
* [jupyter](https://jupyter.org/) (optional for running notebooks)
* [matplotlib](https://matplotlib.org/) (optional for running examples)


You can run examples and tests from the root dir of the repository:

		cd PATH_TO_CROCODDYL
		python examples/talos_arm.py
		python unittest/all.py

If you want to learn about Crocoddyl, take a look at the Jupyter notebooks. Start in the following order.
- [examples/notebooks/unicycle_towards_origin.ipynb](https://gepgitlab.laas.fr/loco-3d/crocoddyl/blob/devel/examples/notebooks/unicycle_towards_origin.ipynb)
- [examples/notebooks/cartpole_swing_up.ipynb](https://gepgitlab.laas.fr/loco-3d/crocoddyl/blob/devel/examples/notebooks/cartpole_swing_up.py)
- [examples/notebooks/manipulator.ipynb](https://gepgitlab.laas.fr/loco-3d/crocoddyl/blob/devel/examples/notebooks/manipulator.ipynb)
- [examples/notebooks/bipedal_walking_from_foot_traj.ipynb](https://gepgitlab.laas.fr/loco-3d/crocoddyl/blob/devel/examples/notebooks/bipedal_walking_from_foot_traj.ipynb)
- [examples/notebooks/introduction_to_crocoddyl.ipynb](https://gepgitlab.laas.fr/loco-3d/crocoddyl/blob/devel/examples/notebooks/introduction_to_crocoddyl.ipynb)


## Citing Crocoddyl

To cite **Crocoddyl** in your academic research, please use the following bibtex lines:
```
@misc{crocoddylweb,
   author = {Carlos Mastalli, Rohan Budhiraja and Nicolas Mansard and others},
   title = {Crocoddyl: a fast and flexible optimal control library for robot control under contact sequence},
   howpublished = {https://gepgitlab.laas.fr/loco-3d/crocoddyl/wikis/home},
   year = {2019}
}
```

and the following one for the reference to the paper introducing **Crocoddyl**:
```
@unpublished{mastalli2020crocoddyl,
  author={Mastalli, Carlos and Budhiraja, Rohan and Merkt, Wolfgang and Saurel, Guilhem and Hammoud, Bilal
  and Naveau, Maximilien and Carpentier, Justin and Vijayakumar, Sethu and Mansard, Nicolas},
  title={{Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control}},
  year={2020}
}
```

The rest of the publications describes different component of **Crocoddyl**:


### Publications
- C. Mastalli et al. [Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control](https://cmastalli.github.io/publications/crocoddyl20unpub.html), pre-print, 2020
- R. Budhiraja, J. Carpentier, C. Mastalli and N. Mansard. [Differential Dynamic Programming for Multi-Phase Rigid Contact Dynamics](https://cmastalli.github.io/publications/mddp18.html), IEEE RAS International Conference on Humanoid Robots (ICHR), 2018
- Y. Tassa, N. Mansard, E. Todorov. [Control-Limited Differential Dynamic Programming](https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf), IEEE International Conference on Automation and Robotics (ICRA), 2014
- R. Budhiraja, J. Carpentier and N. Mansard. [Dynamics Consensus between Centroidal and Whole-Body Models for Locomotion of Legged Robots](https://hal.laas.fr/hal-01875031/document), IEEE International Conference on Automation and Robotics (ICRA), 2019


## Questions and Issues

You have a question or an issue? You may either directly open a [new issue](https://gepgitlab.laas.fr/loco-3d/crocoddyl/issues) or use the mailing list <crocoddyl@laas.fr>.

