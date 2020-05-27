# Overview {#index}
<!--
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Author: Carlos Mastalli, Rohan Budhiraja, Nicolas Mansard
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
-->

\section OverviewIntro What is Crocoddyl?

<img align="right" src="https://i.imgur.com/o2LfbDq.gif" width="250" padding="10"/>

**[Crocoddyl](https://cmastalli.github.io/publications/crocoddyl20icra.html)** is an optimal control library for robot control under contact sequence.
Its solvers are based on novel and efficient Differential Dynamic Programming (DDP) algorithms.
**Crocoddyl** computes optimal trajectories along with optimal feedback gains.
It uses **[Pinocchio](https://github.com/stack-of-tasks/pinocchio)** for fast computation of robots dynamics and their analytical derivatives.

**Crocoddyl** is open-source, mostly written in C++ with Python bindings, and distributed under the BSD licence.
It is one of the most efficient libraries for computing the optimal control with particular enphasis to contact dynamics.

**Crocoddyl** is versatible:

 * various optimal control solvers (DDP, FDDP, BoxDDP, etc) - single and multi-shooting methods
 * analytical and sparse derivatives via **[Pinocchio](https://github.com/stack-of-tasks/pinocchio)**
 * Euclidian and non-Euclidian geometry friendly (handle geometrical systems)
 * handle autonomous and nonautomous dynamical systems
 * numerical differentiation support
 * automatic differentiation support

**Crocoddyl** is efficient and flexible:

 * cache friendly,
 * multi-thread friendly
 * Python bindings (including models and solvers abstractions)
 * C++ 98/11/14/17/20 compliant
 * extensively tested
 * automatic code generation support

<table >
  <tr>
    <td align="left"><img src="https://cmastalli.github.io/assets/img/publications/highly_dynamic_maneuvers.png" width="700"/></td>
    <td align="right"><img src="https://i.imgur.com/RQR2Ovx.gif" width="500"/> <img src="https://i.imgur.com/kTW0ePh.gif" width="500"/></td>
  </tr>
</table>


In this documentation, you will find the usual description of the library functionalities, a quick tutorial to catch over the mathematics behind the implementation, a bunch of examples about how to implement optimal control problems and a set of practical exercices for beginners.


\section OverviewInstall How to install Crocoddyl?

Crocoddyl is best installed from APT packaging on Ubuntu 16.04 and 18.04, from our repository.
<-- On Mac OS X, we support the installation of Pinocchio through the Homebrew package manager. -->
On systems for which binaries are not provided, installation from source should be straightforward.
Every release is validated in the main Linux distributions and Mac OS X.

<!--The full installation procedure can be found on the Github Pages of the project:
http://stack-of-tasks.github.io/pinocchio/download.html.-->

\section OverviewSimple Simplest example with compilation command

We start with a simple optimal control formulation to reach a goal position give the end-effector.

<table class="manual">
  <tr>
    <th>arm_manipulation.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include arm_manipulation.py
    </td>
  </tr>
</table>

You can run this script as:

\code python arm_manipulation.py \endcode

\subsection OverviewSimpleExplain Explanation of the program

This program loads a robot model through example-robot-data, creates a set of action models per node, configures different cost functions, and solves the optimal control problem with our Differential Dynamic Programming (DDP) solver.

The first paragraph describes the optimal control problem that we want to solve "reaching-goal task with the Talos arm".
We have developed custom differential action model (action model in continuous time) and its analytical derivatives for forward dynamics without contact.
This differential action model is called `DifferentialActionModelFreeFwdDynamics`.
We use an Euler sympletic integrator to convert the differential action model into an action model.
Note that our solvers use action model.

To handle the differential manifold of the configuration space, we have developed a state multibody class which can be used with any robotics problem.
We create two kind of action models in this example: running and terminal models.
In the terminal model we include the desired goal cost function.
For the running models, we also include regularization terms that are important for well posing the optimization problem.

We create a trajectory with 250 nodes using the `ShootingProblem` class.
We decouple problem formulation and resoltion through this class.
Note that in Crocoddyl we have the freedom to specialized each node with different cost functions, dynamics and constraints.

After that, we create our DDP solver and define a set of callbacks for analysis: display of iteration in Gepetto viewer, and print of iteration values.
Finally, we have created custom plot functions for easily check of results.


\section OverviewPython About Python wrappings

Crocoddyl is written in C++, with a full template-based C++ API, for code generation and automatic differentiation purposes. All the functionalities are available in C++. Extension of the library should be preferably in C++.

C++ interface is efficient and easy to code. It depends on virtualization with a minimal templatization for codegen and autodiff.
The Python API mirrors quite closely the C++ interface. The greatest difference is that the C++ interface is proposed using Eigen objects for matrices and vectors, that are exposed as NumPy matrices in Python.

When working with Crocoddyl, we often suggest to first prototype your ideas in Python.
Models, costs and constraints can be derived in Python as well; making the whole prototyping process quite easy.
Both the auto-typing and the scripting make it much faster to develop. Once you are happy with your prototype, then translate it in C++ while binding the API to have a mirror in Python that you can use to extend your idea.
Currently, the codegen and autodiff are only available in C++ interface. However, it is in our plan to deploy in Python too. Your contributions are welcome!

\section OverviewCite How to cite Crocoddyl

Happy with Crocoddyl? Please cite us with the following format.

### Easy solution: cite our open access paper
The following is the preferred way to cite Crocoddyl or the feasibility-drive DDP solver.
The paper is publicly available in ([ArXiv](https://arxiv.org/abs/1909.04947 "Carlos Mastalli et al - Crocoddyl paper")).

\include mastalli-icra20.bib


\section OverviewConclu Where to go from here?

This documentation is mostly composed of several examples and tutorials for newcomers, along with a technical documentation and a reference guide. If you want to make sure Crocoddyl matches your needs, you may first want to check the list of features. Several examples in Python will then directly give you the keys to implement the most classical applications based on a Crocoddyl library. For nonexperts, we also provide the main mathematical fundamentals of optimal control. A long tutorial in Python contains everything you need if you are not a Python expert and want to start with Crocoddyl. This tutorial was first written as course material for the MEMMO winter school.

That's it for beginners. We then give an overview of the technical choices we made to write the library and make it efficient. A description of the benchmarks we did to test the library efficiency is also provided.
