
<img align="right" src="https://i.imgur.com/o2LfbDq.gif" width="25%"/>

Contact RObot COntrol by Differential DYnamic programming Library (crocoddyl)
===============================================


<table >
  <tr>
    <td align="left"><img src="https://cmastalli.github.io/assets/img/publications/highly_dynamic_maneuvers.png" width="10000"/></td>
    <td align="right"><img src="https://i.imgur.com/RQR2Ovx.gif"/> <img src="https://i.imgur.com/kTW0ePh.gif"/></td>
  </tr>
</table>


## <img align="center" height="20" src="https://i.imgur.com/vAYeCzC.png"/> Introduction

**[Crocoddyl](https://cmastalli.github.io/publications/crocoddyl20unpub.html)** is an optimal control library for robot control under contact sequence.
Its solver is based on an efficient Differential Dynamic Programming (DDP) algorithm.
**Crocoddyl** computes optimal trajectories along to optimal feedback gains.
It uses **Pinocchio** for fast computation of robot dynamics and its analytical derivatives.

The source code is released under the [BSD 3-Clause license](LICENSE).

**Authors:** [Carlos Mastalli](https://cmastalli.github.io/) and Rohan Budhiraja <br />
**Instructors:** Nicolas Mansard <br />
**With additional support from the Gepetto team at LAAS-CNRS and MEMMO project. For more details see Section Credits**

[![License BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg?style=flat)](https://tldrlegal.com/license/bsd-3-clause-license-%28revised%29#fulltext)
[![CI build status](https://travis-ci.org/loco-3d/crocoddyl.svg?branch=master)](https://travis-ci.org/loco-3d/crocoddyl)
[![pipeline status](https://gepgitlab.laas.fr/loco-3d/crocoddyl/badges/devel/pipeline.svg)](https://gepgitlab.laas.fr/loco-3d/crocoddyl/pipelines?ref=devel)
[![coverage report](https://gepgitlab.laas.fr/loco-3d/crocoddyl/badges/devel/coverage.svg)](https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/devel/coverage/)
[![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/loco-3d/crocoddyl)](https://gepgitlab.laas.fr/loco-3d/crocoddyl/-/tags)
[![GitHub repo size](https://img.shields.io/github/repo-size/loco-3d/crocoddyl)](https://img.shields.io/github/repo-size/loco-3d/crocoddyl)
[![contributors](https://img.shields.io/github/contributors/loco-3d/crocoddyl)](https://gepgitlab.laas.fr/loco-3d/crocoddyl/-/graphs/master)

[![GitHub Release Date](https://img.shields.io/github/release-date/loco-3d/crocoddyl)](https://img.shields.io/github/release-date/loco-3d/crocoddyl)
[![GitHub last commit](https://img.shields.io/github/last-commit/loco-3d/crocoddyl)](https://img.shields.io/github/last-commit/loco-3d/crocoddyl)

If you want to follow the current developments, you can directly refer to the [devel branch](https://gepgitlab.laas.fr/loco-3d/cddp/tree/devel).


## <img align="center" height="20" src="https://i.imgur.com/x1morBF.png"/> Installation
**Crocoddyl** can be easily installed on various Linux (Ubuntu, Fedora, etc.) and Unix distributions (Mac OS X, BSD, etc.).

## Crocoddyl features
**Crocoddyl** is versatible:

 * various optimal control solvers (DDP, FDDP, BoxDDP, etc) - single and multi-shooting methods
 * analytical and sparse derivatives
 * geometrical systems friendly (with SE(3) manifold support)
 * handle autonomous and nonautomous dynamical systems
 * numerical differentiation support
<!-- * automatic differentiation support -->

**Crocoddyl** is efficient and flexible:

 * cache friendly,
 * multi-thread friendly
 * Python bindings (including models and solvers abstractions)
 * C++98/11/14/17/20 compliant
 * extensively tested
 <!-- * automatic code generation support -->
 

### Installation through robotpkg

You can install this package through robotpkg. robotpkg is a package manager tailored for robotics softwares.
It greatly simplifies the release of new versions along with the management of their dependencies.
You just need to add the robotpkg apt repository to your sources.list and then use `sudo apt install robotpkg-py27-crocoddyl` (or `py3X` for python 3.X, depending on your system):

If you have never added robotpkg as a softwares repository, please follow first the instructions from 1 to 3; otherwise, go directly to instruction 4.
Those instructions are similar to the installation procedures presented in [http://robotpkg.openrobots.org/debian.html](http://robotpkg.openrobots.org/debian.html).

1. Add robotpkg as source repository to apt:

```bash
sudo tee /etc/apt/sources.list.d/robotpkg.list <<EOF
deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -sc) robotpkg
deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg
EOF
```

2. Register the authentication certificate of robotpkg:

```bash
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
```

3. You need to run at least once apt update to fetch the package descriptions:

```bash
sudo apt-get update
```

4. The installation of Crocoddyl:

```bash
sudo apt install robotpkg-py27-crocoddyl # for Python 2

sudo apt install robotpkg-py35-crocoddyl # for Python 3
```

Finally you will need to configure your environment variables, e.g.:

```bash
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/openrobots/lib/python2.7/site-packages:$PYTHONPATH
```


### Building from source

**Crocoddyl** is c++ library with Python bindings for versatile and fast prototyping. It has the following dependencies:

* [pinocchio](https://github.com/stack-of-tasks/pinocchio)
* [example-robot-data](https://gepgitlab.laas.fr/gepetto/example-robot-data) (optional for examples, install Python loaders)
* [gepetto-viewer-corba](https://github.com/Gepetto/gepetto-viewer-corba) (optional for display)
* [jupyter](https://jupyter.org/) (optional for notebooks)
* [matplotlib](https://matplotlib.org/) (optional for examples)


You can run examples, unit-tests and benchmarks from your build dir:

```bash
cd build
make test
make -s examples-quadrupedal_gaits INPUT="display plot" # enable display and plot
make -s benchmarks-quadrupedal_gaits INPUT="100 walk" # number of trials ; type of gait
```

Alternatively, you cansee the 3D result and/or graphs of your run examples, you can use

```bash
export CROCODDYL_DISPLAY=1
export CROCODDYL_PLOT=1
```

If you want to learn about Crocoddyl, take a look at the Jupyter notebooks. Start in the following order.
- [examples/notebooks/unicycle_towards_origin.ipynb](https://gepgitlab.laas.fr/loco-3d/crocoddyl/blob/devel/examples/notebooks/unicycle_towards_origin.ipynb)
- [examples/notebooks/cartpole_swing_up.ipynb](https://gepgitlab.laas.fr/loco-3d/crocoddyl/blob/devel/examples/notebooks/cartpole_swing_up.py)
- [examples/notebooks/arm_manipulation.ipynb](https://gepgitlab.laas.fr/loco-3d/crocoddyl/blob/devel/examples/notebooks/arm_manipulation.ipynb)
- [examples/notebooks/bipedal_walking.ipynb](https://gepgitlab.laas.fr/loco-3d/crocoddyl/blob/devel/examples/notebooks/bipedal_walking.ipynb)
- [examples/notebooks/introduction_to_crocoddyl.ipynb](https://gepgitlab.laas.fr/loco-3d/crocoddyl/blob/devel/examples/notebooks/introduction_to_crocoddyl.ipynb)


## Citing Crocoddyl

To cite **Crocoddyl** in your academic research, please use the following bibtex lines:
```tex
@unpublished{mastalli2020crocoddyl,
  author={Mastalli, Carlos and Budhiraja, Rohan and Merkt, Wolfgang and Saurel, Guilhem and Hammoud, Bilal
  and Naveau, Maximilien and Carpentier, Justin and Vijayakumar, Sethu and Mansard, Nicolas},
  title={{Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control}},
  year={2020}
}
```
and the following one to reference this website:
```tex
@misc{crocoddylweb,
   author = {Carlos Mastalli, Rohan Budhiraja and Nicolas Mansard and others},
   title = {Crocoddyl: a fast and flexible optimal control library for robot control under contact sequence},
   howpublished = {https://gepgitlab.laas.fr/loco-3d/crocoddyl/wikis/home},
   year = {2019}
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


## Credits

The following people have been involved in the development of **Crocoddyl**:

- [Carlos Mastalli](https://cmastalli.github.io/) (University of Edinburgh): main developer and manager of the project
- [Nicolas Mansard](http://projects.laas.fr/gepetto/index.php/Members/NicolasMansard) (LAAS-CNRS): project instructor
- [Rohan Budhiraja](https://scholar.google.com/citations?user=NW9Io9AAAAAJ) (LAAS-CNRS): features extension
- [Justin Carpentier](https://jcarpent.github.io/) (INRIA): efficient analytical rigid-body dynamics derivatives
- [Maximilien Naveau](https://scholar.google.fr/citations?user=y_-cGlUAAAAJ&hl=fr) (MPI): unit-test support
- [Guilhem Saurel](http://projects.laas.fr/gepetto/index.php/Members/GuilhemSaurel) (LAAS-CNRS): continuous integration and deployment
- [Wolfgang Merkt](http://www.wolfgangmerkt.com/research/) (University of Edinburgh): feature extension and debugging
- [Josep Martí Saumell](https://www.iri.upc.edu/staff/jmarti) (UPC): feature extension
- [Bilal Hammoud](https://scholar.google.com/citations?hl=en&user=h_4NKpsAAAAJ) (MPI): features extension


## Acknowledgments

The development of **Pinocchio** is supported by the [EU MEMMO project](http://www.memmo-project.eu/), the [Gepetto team](http://projects.laas.fr/gepetto/) [@LAAS-CNRS](http://www.laas.fr), and the [Statistical Machine Learning and Motor Control Group](http://wcms.inf.ed.ac.uk/ipab/slmc) [@University of Edinburgh](https://www.edinburgh-robotics.org/).
