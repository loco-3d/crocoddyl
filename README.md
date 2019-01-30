Contact RObot COntrol by Differential DYnamic programming Library (crocoddyl)
===============================================

## <img align="center" height="20" src="https://i.imgur.com/vAYeCzC.png"/> Introduction

**Crocoddyl** is an optimal control library for robot control under contact sequence. Its solver is based on an efficient Differential Dynamic Programming (DDP) algorithm. **Crocoddyl** computes optimal trajectories along to optimal feedback gains. It uses **Pinocchio** for fast computation of robot dynamics and its analytical derivatives.

The source code is released under the [BSD 3-Clause license](LICENSE).

**Authors:** Carlos Mastalli and Rohan Budhiraja <br />
**Instructors:** Justin Carpentier and Nicolas Mansard <br />
**With additional support from the Gepetto team at LAAS-CNRS.**

[![License BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg?style=flat)](https://tldrlegal.com/license/bsd-3-clause-license-%28revised%29#fulltext)
[![pipeline status](https://gepgitlab.laas.fr/loco-3d/cddp/badges/master/pipeline.svg)](https://gepgitlab.laas.fr/loco-3d/cddp/commits/master)
[![coverage report](https://gepgitlab.laas.fr/loco-3d/cddp/badges/master/coverage.svg)](https://gepgitlab.laas.fr/loco-3d/cddp/commits/master)

If you want to follow the current developments, you can directly refer to the [devel branch](https://gepgitlab.laas.fr/loco-3d/cddp/tree/devel).


## <img align="center" height="20" src="https://i.imgur.com/x1morBF.png"/> Installation
**Crocoddyl** has the following dependecies:

* boost (unit_test_framework)
* eigen3
* [pinocchio](https://github.com/stack-of-tasks/pinocchio)

To install eigen3 on Ubuntu you can use apt-get:
  sudo apt-get install libeigen3-dev

To install [pinocchio](https://github.com/stack-of-tasks/pinocchio) follow the instructions on its [website](https://stack-of-tasks.github.io/pinocchio/download.html).


## <img align="center" height="20" src="http://www.pvhc.net/img205/oohmbjfzlxapxqbpkawx.png"/> Publications
- R. Budhiraja, J. Carpentier, C. Mastalli and N. Mansard. [Differential Dynamic Programming for Multi-Phase Rigid Contact Dynamics](https://hal.archives-ouvertes.fr/hal-01851596/document), International Conference on Humanoid Robots (ICHR), 2018
- R. Budhiraja, J. Carpentier and N. Mansard. [Dynamics Consensus between Centroidal and Whole-Body Models for Locomotion of Legged Robots](https://hal.laas.fr/hal-01875031/document), Submitted to the International Conference on Automation and Robotics (ICRA), 2019
