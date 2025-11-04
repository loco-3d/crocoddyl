# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

* Introduced safe difference and integration functions for states in https://github.com/loco-3d/crocoddyl/pull/1448
* ROS: jrl_cmakemodules dependency + kilted CI in https://github.com/loco-3d/crocoddyl/pull/1441

## [3.1.0] - 2025-10-02

* Allow use of BUILD_STANDALONE_PYTHON_INTERFACE CMake option in https://github.com/loco-3d/crocoddyl/pull/1434
* Migrated notebooks to Meshcat in https://github.com/loco-3d/crocoddyl/pull/1404
* Updated the rotation transpose in contact wrench cone to re-use the same object to avoid additional transpose operations. https://github.com/loco-3d/crocoddyl/pull/1343
* Switched to a multiply operation instead of divide for efficiency. https://github.com/loco-3d/crocoddyl/pull/1343
* Fix Python bindings of action getters in https://github.com/loco-3d/crocoddyl/pull/1374
* Created independent Python modules associated to each scalar in https://github.com/loco-3d/crocoddyl/pull/1372
* Introduced CodeGen and AutoDiff in Python + pre-compiled headers in https://github.com/loco-3d/crocoddyl/pull/1370
* Introduced explicit template instantiation in https://github.com/loco-3d/crocoddyl/pull/1367
* Fixed error message building crocoddyl.launch crocoddyl.rviz as Python files in https://github.com/loco-3d/crocoddyl/pull/1369
* Created Python bindings for (diff)-action getters in https://github.com/loco-3d/crocoddyl/pull/1362
* Enabled casting and multi-scalar Python bindings in https://github.com/loco-3d/crocoddyl/pull/1346
* Compute LU decomposition using permutationP in https://github.com/loco-3d/crocoddyl/pull/1351
* Force DifferentialActionDataContactInvDynamics to update df_du for all contacts in  https://github.com/loco-3d/crocoddyl/pull/1348
* Add a nix shortcut start jupyter lab, with eg. `nix run .#jupyter in https://github.com/loco-3d/crocoddyl/pull/1396

## [3.0.1] - 2025-03-21

* Add install version in https://github.com/loco-3d/crocoddyl/pull/1355
* Removed absolute path for boost library in https://github.com/loco-3d/crocoddyl/pull/1353
* Fixed checking of positive semi-define condition in LQR problems in https://github.com/loco-3d/crocoddyl/pull/1352

## [3.0.0] - 2025-03-19

* :warning: BREAKING: replaced boost shared pointers by std ones in https://github.com/loco-3d/crocoddyl/pull/1339
* :warning: BREAKING: require pinocchio >= 3.4.0

## [2.2.0] - 2025-02-10

* Changed return policy in std::vector of Eigen's vector and matrices to be compliant with Pinocchio in https://github.com/loco-3d/crocoddyl/pull/1338
* Prevent users to improperly setting residual references in https://github.com/loco-3d/crocoddyl/pull/1332
* Fixed the inequality constraints' feasibility computation by incorporating bounds into the calculation in https://github.com/loco-3d/crocoddyl/pull/1307
* Improved the action factory used for unit testing in https://github.com/loco-3d/crocoddyl/pull/1300
* Ignore ruff issues in ipython notebook files in https://github.com/loco-3d/crocoddyl/pull/1297
* Improved efficiency for computing impulse-dynamics derivatives in https://github.com/loco-3d/crocoddyl/pull/1294
* Fixed bug of wrench cone fields not being updated with setters in https://github.com/loco-3d/crocoddyl/pull/1274
* Replaced parent by parentJoint (which was introduced in Pinocchio 3) in https://github.com/loco-3d/crocoddyl/pull/1271
* Introduced the notion of terminal dimension, residuals and constraints in https://github.com/loco-3d/crocoddyl/pull/1269
* General clean ups about std::size_t in https://github.com/loco-3d/crocoddyl/pull/1265
* Fixed issues in LQR extensions in https://github.com/loco-3d/crocoddyl/pull/1263
* Computed dynamic feasibility everytime in https://github.com/loco-3d/crocoddyl/pull/1262
* Extend LQR actions in https://github.com/loco-3d/crocoddyl/pull/1261
* Print log with grad absolute + updated logs with Pinocchio 3 in https://github.com/loco-3d/crocoddyl/pull/1260

## [2.1.0] - 2024-05-31

* Updated black + isort + flake8 to ruff in https://github.com/loco-3d/crocoddyl/pull/1256
* Exported version for Python in https://github.com/loco-3d/crocoddyl/pull/1254
* Added pinocchio 3 preliminary support in https://github.com/loco-3d/crocoddyl/pull/1253
* Updated CMake packaging in https://github.com/loco-3d/crocoddyl/pull/1249
* Fixed ruff reported error in https://github.com/loco-3d/crocoddyl/pull/1248
* Fixed yapf reported errors in https://github.com/loco-3d/crocoddyl/pull/1238
* Tested Python stubs in Conda CI in https://github.com/loco-3d/crocoddyl/pull/1228
* Fixed Rviz display in https://github.com/loco-3d/crocoddyl/pull/1227
* Improved CI, updated cmake and fixed launch file in https://github.com/loco-3d/crocoddyl/pull/1220
* Introduced a Rviz display in https://github.com/loco-3d/crocoddyl/pull/1216
* Enabled display of thrust and simplied displayers code in https://github.com/loco-3d/crocoddyl/pull/1215
* Introduced floating base thruster actuation model in https://github.com/loco-3d/crocoddyl/pull/1213
* Fixed quadruped and biped examples in https://github.com/loco-3d/crocoddyl/pull/1208
* Fixed terminal computation in Python models in https://github.com/loco-3d/crocoddyl/pull/1204
* Fixed handling of unbounded values for `ActivationBounds` in https://github.com/loco-3d/crocoddyl/pull/1191

## [2.0.2] - 2023-12-07

* Added nu, ng, and nh setters for Python bindings in https://github.com/loco-3d/crocoddyl/pull/1192
* Added CHANGELOG.md in https://github.com/loco-3d/crocoddyl/pull/1188
* Supported nu==0 in actuation models in https://github.com/loco-3d/crocoddyl/pull/1188
* Included Python bindings for Crocoddyl exceptions by @cmastalli in https://github.com/loco-3d/crocoddyl/pull/1186
* Updated cmake submodule update by @jcarpentier in https://github.com/loco-3d/crocoddyl/pull/1186
* Fixed getters for contraints bounds by skleff1994 in https://github.com/loco-3d/crocoddyl/pull/1180
* Extended solver abstract and callbacks for arbitrary solvers by @cmastalli in https://github.com/loco-3d/crocoddyl/pull/1179
* Fixed the check of pair_id in collision residual by @ArthurH91 in https://github.com/loco-3d/crocoddyl/pull/1178
* Exploited control-residual structure when computing Lu, Luu by @cmastalli in https://github.com/loco-3d/crocoddyl/pull/1176
* Added LWA fram convention and introduced different axis for 1d contacts by @skleff1994 in https://github.com/loco-3d/crocoddyl/pull/1172
* Python bindings for setting control bounds by @cmastalli in https://github.com/loco-3d/crocoddyl/pull/1171
* Fixed missed scalar in cost sum and activation data by @cmastalli in https://github.com/loco-3d/crocoddyl/pull/1165
* Added actuation unit tests by @cmastalli in https://github.com/loco-3d/crocoddyl/pull/1161
* Introduced method for obtaining the dimension of floating-bases by @cmastalli in https://github.com/loco-3d/crocoddyl/pull/1160
* Fixed set_reference in state residual by @cmastalli in https://github.com/loco-3d/crocoddyl/pull/1158
* Enabled CONDA CI jobs with CppADCodeGen by @cmastalli in https://github.com/loco-3d/crocoddyl/pull/1156
* Added other CI jobs by @cmastalli in https://github.com/loco-3d/crocoddyl/pull/1152
* Fixed compiltation issue when building with CppADCodeGen by @cmastalli in https://github.com/loco-3d/crocoddyl/pull/1151
* Fixed include order used in frames.cpp by @ManifoldFR in https://github.com/loco-3d/crocoddyl/pull/1150

## [2.0.1] - 2023-06-17

* Fixed notebooks
* Fixed build on aarch64-linux
* Fixed CMake for OpenMP with conda
* Fixed lints
* Updated for example-robot-data 4.0.7
* Added support for 18.04 back: CMake 3.10, Boost 1.65, Python 3.6

## [2.0.0] - 2023-05-13

* Changed stopping criteria to better evaluate the converge criteria
* Extended numdiff routines to compute second-order derivatives + other minor improvements
* Improved the overall project documentation (still many things to be done)
* Added collision residual unit tests and missing Python bindings
* Allowed accuracy configuration of verbose callback
* Closed gaps once feasibility is achieved in FDDP
* Supported different Python versions
* Added procedure to check example log files
* Added support to M1 apple chip + fixed compilation issues with Clang
* Fixed a small issue in the solver's Armijo condition
* Added shareMemory unit tests
* Introduced the notion of resizing data in solvers
* Added unit test for impulse actions
* Deprecated set_id functions in contact residuals
* Supported Ipopt solver with Python bindings
* Extended (diff)-action API to handle arbitrary constraints + unit tests
* Introduce the notion of ConstraintManager and used it in diff-actions + unit tests + integrated action support
* Added the HyQ robot for extra unit testing
* Updated doxygen documentation in many parts of the project
* Used std::set for storing cost/constraint activation status
* Fixed small issues in terminal-node computations
* Supported Meshcat display of contact forces, friction cone and foot-swing trajectories
* Introduced the notion of equality constraint feasibility in solvers
* Created a data collector to store joint effort and accelerations
* Created joint-effort/acceleration residuals + unit tests
* Created inverse-dynamics action models + unit tests
* Added the SolverIntro which handles inverse-dynamics OC problems + unit tests
* Added invdyn examples and log files + cleaned up filenames
* Enabled copyable for various objects used in Python
* Supported different contact/impulse frames + unit test + changes in example/logfiles.
* Removed deprecated FrameXX code.
* Updated readme file.
* Updated CMake configuration
* Updated numpy usage

## [1.9.0] - 2022-03-03

* Introduced the control parametrization notion and three polynomial parametrization (PolyZero, PolyOne, PolyTwoRK)
* Improved the documentation of the actuation model and especially the floating-base actuation
* Improved the efficiency of the RK integrator (added to the benchmarks)
* Improved the documentation of Euler and RK integrators
* Added unittests for checking the analytical derivatives of the contact forces/impulses
* Computed the dynamic feasibility in solvers (also print this relevant information)
* Improved the documentation of solvers
* Removed dynamic memory allocation in ContactModel3D
* Added the notion of terminal (calc, calcDiff) computations (fixed some tiny inaccuracies in the problem formulation + reduced useless computations)
* Added a class to easily profile the computation cost of any block code (included relevant blocks in the solvers and shooting problem)
* Fixed error in the documentation of ActivationModel2NormBarrier
* Added the 2d contact
* Added method to resized solver data (e.g., it is not needed to allocate the biggest control dimension)
* Fixed issue in the Gepetto viewer display
* Improved the CI and fixed a few errors that appears in unittests code compiled with clang
* Removed dynamics memory allocation in LQR action and CostModelResidual
* Removed Travis buildfarm and substituted by ROS one
* Used std::set for contact/impulse active/inactive set (added bindings)
* Added Python bindings to be able to set state dimensions from a Python derived class.
* Added Python bindings of StateNumDiff class

## [1.8.1] - 2021-08-01

* Fixed Vx computation
* Fixed memory allocations
* Fixed linkage of the python library
* Deprecated FramXX in constructors and frames.hpp
* Cleaned up code

## [1.7.0] - 2021-05-05

* Updated the examples based on new API in example-robot-data
* Removed reference in std::sized_t (and other primitives)
* Improved computation and handled richer conditions in friction and wrench cone (e.g., inner/outer apprx. in wrench cone, and rotation matrix in friction cone)
* Added more unit-tests for cones
* Included the CoP support notion
* Updated minimal version of EigenPy as it fixes a bug with 4x6 matrices
* Developed a gravity-based cost function for both free and in contact conditions (included its unit-test code)
* Added assignment operator in FrameXXX structures
* Replaced isMuchSmallerThan by isZero in all the unit-tests
* Enabled free-flyer joint in full actuation model
* Improved multicopter actuation API + unit-tests
* Exposed in Python ActionModelUnicyle::dt_
* Fixed multithreading support: running the correct number of threads
* Enabled that the user can set the number of threads (also in Python)
* Added publication list and updated README file
* Registered in Python the shared pointers of all the model classes
* Fixed Meshcat visualizer after update
* Added Github Action CI with ROS dependency resolution
* Improved efficiency of backward pass by defining properly RowMajor matrices
* Activated all warnings and Werror
* Improved documentation

## [1.6.0] - 2021-02-01

* Refactored the Cost API with breaking compatibility (cost depends on state abstract, not multibody state)
* Fixed issue in c++98 compatibility
* Added shared_ptr registers for solver classes (Python bindings)
* Initialized missed data in SolverQP (not really producing a bug)
* Fixed issue with FrameXXX allocators (Python bindings)
* Created aligned std vectors for FrameXXX (Python bindings)
* Used the proper nu-dimension in shooting problem and solvers
* Doxygen documentation of smooth-abs activation
* Renamed the activation model: smooth-abs to smooth-1norm (deprecated old names)
* Added the smooth-2norm activation model with Python bindings
* Updated README file with Credits and committee information
* Added throw_pretty in Python bindings of action models (checks x,u dimensions)
* Improved the documentation (doxygen, docstrings), and fixed grammar, of various classes
* Cleaned up a few things related with cost classes
* Cleaned up unnecessary typedef in cost models
* Extended the differential action model for contacts to handle any actuation model (not just floating-base derived ones)
* Added conda support
* Added the quadratic-flat activation models
* Fixed issue with gepetto viewer display (appearing in some OS)
* Added contact/impulse action unit tests
* Added contact/impulse cost unit tests
* Added a proper gap threshold (it was too big and created different behavior in feasibility-driven solvers)
* Improved the copyright starting year for many files

## [1.5.0] - 2020-09-23

* Improved and cleaned up the bench-marking code for code-generation
* Fixed bug for computing quasicStatic torques under inactive contacts
* Added unit-test code that disables contacts
* Created CoP cost with Python bindings and unit-test
* Multi-threading support for quasicStatic computation in shooting problem
* Modifications in Travis CI + included Gepgitlab CI
* Created contact wrench cone (CWC) cost with Python bindings and unit-test
* Created RK4 integrator with Python bindings and unit-test
* Throw exception for setting candidates xs/us in solvers
* Fixed a few spelling errors in the docstring documentation
* Exposed the KKT solver in Python
* Checked the dimension of the warm-start vectors for xs and us
* Cleaned up some part of the Python code that were using Numpy Matrix (now Numpy Array!)
* Created the 2d contact
* Checked the feasibility by the gap values
* Created the Crocoddyl logo + integrated in the README file

## [1.4.0] - 2020-08-05

* Allowed different values of nu in action models
* Adapted the contact force cost to accept 3d contacts.
* Added CppAD support to (weighted) quadratic barrier activation
* Added the contact impulse cost
* Added contact CoP cost,
* fixes:
* Fixed display of contact forces in Gepetto viewer
* Fixed a bug in the definition of the terminal node in updateNode (defined in ShootingProblem)
* Fixed memory allocation produced by Python binds.
* Fixed printed error message in ShootingProblem
* and improvements:
* Added documentation of the differential action model
* Deprecated legacy functions to define reference in cost functions.
* Set to zero inactive contact forces for correctness
* Simplified FDDP code by using calcDiff from DDP solver
* Improved the display of benchmark timings
* Improved the documentation of contact force cost
* Improved few notebooks and added an extra one
* Removed legacy Python unittest

## [1.3.0] - 2020-05-25

* Minor improvement in state-base class (enabling limits once we define only one component)
* Added functions to print the active/inactive status of costs, contacts and impulses
* Running computations in the contact / impulse dynamics using the active number of contacts / impulses, respectively
* Used NumPy Array in examples / notebooks
* Initial integration of Meshcat viewer and used in notebooks
* Improved efficiency of Jdiff, Jintegrate in multibody state
* Added an extra operator in Jintegrate signature (state classes)
* Added an extra function parallel transport in the state class (i.e. JintegrateTransport)
* Added functions to retrieve the inactive costs, contacts and impulses
* Fixed target_link_libraries use
* Improved efficiency of few activation models
* Added the action-base class for code generation + unittest
* Added codegen bench for 4 dof arm and biped
* Renamed all bench files
* Updated dependency versions for Pinocchio and example-robot-data
* Fixed bug in the impulse dynamics that appears in multiple threads
* Fixed various issues regarding data alignment
* Added doxygen file with the documentation state-base class
* Fixed an issue when we updates reference in frame-placement and frame-rotation costs
* Fixed an issue with Python3 compatibility in few examples
* Proper display of friction cones + do not display when the contact / impulse is inactive/
* Extended the computation of impulses derivatives
* Added a friction cost for impulse dynamics

## [1.2.1] - 2020-04-21

* Fixed backup files inside notebook folder
* Fixed an issue with the friction cone display
* Added quadcopter actuation model

## [1.2.0] - 2020-04-03

* Templatized all the classes and structures with the scalar (for codegen and autodiff)
* Fixed a bug in the formulation for the quadrupedal problem (state bounds)
* Reduced the compilation time in Python bindings (re-structured the code)
* Fixed error in the expected improvement computation for terminal action in SolverFDDP
* Added unittest code for cost classes (included cost factory)
* Reorganized the various factories used for c++ unittesting
* Developed the pinocchio model factory for c++ unittesting
* Added unittest code for contact classes (included the contact factory)
* Described cost items through shared_ptr
* Described contact items through shared_ptr
* Described impulse items through shared_ptr
* Fixed Gauss-Newton approach for cost num-diff
* Added contact num-diff class
* Used virtual keyword in declaration of derived functions
* Added the unittest code for free forward dynamics action model
* Added the unittest code for contact forward dynamics action model
* Included the cost status in cost sum for global memory allocation
* Included the contact status in cost sum for global memory allocation
* Included the impulse status in cost sum for global memory allocation
* Removed duplication function for retrieving models and datas in solvers (now we do only through shooting problem interface)
* Allowed to write internal data of Numpy-EigenPy objects
* Added a general method for setting and getting cost reference
* Added squashing function abstraction
* Added unittest code for testing cost, contact and impulse status
* Moved to CMake exports

## [1.1.0] - 2020-02-14

* Added few cost functions related to forces and impulses (e.g. friction-cone, com impulse)
* Improved the display tools (friction cone, contact forces and end-effector trajectories)
* Fixed a problem in the printed message by the callback
* Fixed a problem in the box-qp
* Improved the box-ddp
* Added a new solver called box-fddp
* Added extra examples (box-ddp vs box-fddp, taichi task, humanoid manipulation)
* Added a script for automatically updating the log files
* Checked that all examples runs in the CI (for release mode only)
* Improved the quadrupedal and bipedal examples by adding all the constraints
* Improved the efficiency in differential free-forward dynamics
* Added extra setter functions in action models
* Improved efficiency in all solvers (removed extra computation)
* Improved plot functions
* Fixed a bug in few notebooks

## [1.0.0] - 2019-08-30

Initial release

[Unreleased]: https://github.com/loco-3d/crocoddyl/compare/v3.1.0...HEAD
[3.1.0]: https://github.com/loco-3d/crocoddyl/compare/v3.0.1...v3.1.0
[3.0.1]: https://github.com/loco-3d/crocoddyl/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/loco-3d/crocoddyl/compare/v2.2.0...v3.0.0
[2.2.0]: https://github.com/loco-3d/crocoddyl/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/loco-3d/crocoddyl/compare/v2.0.2...v2.1.0
[2.0.2]: https://github.com/loco-3d/crocoddyl/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/loco-3d/crocoddyl/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/loco-3d/crocoddyl/compare/v1.9.0...v2.0.0
[1.9.0]: https://github.com/loco-3d/crocoddyl/compare/v1.8.1...v1.9.0
[1.8.1]: https://github.com/loco-3d/crocoddyl/compare/v1.8.0...v1.8.1
[1.8.0]: https://github.com/loco-3d/crocoddyl/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/loco-3d/crocoddyl/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/loco-3d/crocoddyl/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/loco-3d/crocoddyl/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/loco-3d/crocoddyl/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/loco-3d/crocoddyl/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/loco-3d/crocoddyl/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/loco-3d/crocoddyl/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/loco-3d/crocoddyl/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/loco-3d/crocoddyl/releases/tag/v1.0.0
