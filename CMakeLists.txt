CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

# Set up project properties
SET(PROJECT_NAMESPACE loco-3d)
SET(PROJECT_NAME crocoddyl)
SET(PROJECT_DESCRIPTION "Contact RObot COntrol by Differential DYnamic programming Library (Crocoddyl)")
SET(PROJECT_URL https://github.com/${PROJECT_NAMESPACE}/${PROJECT_NAME})

OPTION(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)

# Check if the submodule cmake have been initialized
SET(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  message(STATUS "JRL cmakemodules not found. Let's fetch it.")
  include(FetchContent)
  FetchContent_Declare(
    "jrl-cmakemodules"
    GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
  FetchContent_MakeAvailable("jrl-cmakemodules")
  FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
ENDIF()

# Include important cmake modules
INCLUDE("${JRL_CMAKE_MODULES}/base.cmake")
INCLUDE("${JRL_CMAKE_MODULES}/boost.cmake")
INCLUDE("${JRL_CMAKE_MODULES}/apple.cmake")
INCLUDE("${JRL_CMAKE_MODULES}/stubs.cmake")

# Print initial message
MESSAGE(STATUS "${PROJECT_DESCRIPTION}, version ${PROJECT_VERSION}")
MESSAGE(STATUS "Copyright (C) 2018-2023 CNRS-LAAS, University of Edinburgh, INRIA")
MESSAGE(STATUS "                        Heriot-Watt University")
MESSAGE(STATUS "All rights reserved.")
MESSAGE(STATUS "Released under the BSD 3-Clause License.")

# Set a default build type to 'Release' if none was specified
SET_DEFAULT_CMAKE_BUILD_TYPE(Release)

SET(DOXYGEN_USE_MATHJAX YES)
SET(DOXYGEN_USE_TEMPLATE_CSS YES)

SET(CXX_DISABLE_WERROR False)
SET(PROJECT_USE_CMAKE_EXPORT True)

# Create different building options
OPTION(ENABLE_VECTORIZATION "Enable vectorization and further processor-related optimizations" OFF)
OPTION(BUILD_PYTHON_INTERFACE "Build the python binding" ON)
OPTION(BUILD_BENCHMARK "Build the benchmark" ON)
OPTION(BUILD_EXAMPLES "Build the examples" ON)
OPTION(BUILD_WITH_IPOPT "Build the IPOPT wrapper." ON)
OPTION(GENERATE_PYTHON_STUBS "Generate the Python stubs associated to the Python library" OFF)

IF(ENABLE_VECTORIZATION)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
ENDIF()

COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
PROJECT(${PROJECT_NAME} ${PROJECT_ARGS})

# If needed, fix CMake policy for APPLE systems
APPLY_DEFAULT_APPLE_CONFIGURATION()

CHECK_MINIMAL_CXX_STANDARD(14 ENFORCE)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/find-external/CppAD/" ${CMAKE_MODULE_PATH})

# Add the different required and optional dependencies
IF(BUILD_PYTHON_INTERFACE)
  ADD_PROJECT_DEPENDENCY(eigenpy 2.8.0 REQUIRED PKG_CONFIG_REQUIRES "eigenpy >= 2.8.0")
ENDIF(BUILD_PYTHON_INTERFACE)
ADD_PROJECT_DEPENDENCY(pinocchio 2.6.3 REQUIRED PKG_CONFIG_REQUIRES "pinocchio >= 2.6.3")
IF(BUILD_EXAMPLES OR BUILD_TESTING OR BUILD_BENCHMARK)
  ADD_PROJECT_DEPENDENCY(example-robot-data 4.0.0 REQUIRED PKG_CONFIG_REQUIRES "example-robot-data >= 4.0.0")
ELSE()
  ADD_OPTIONAL_DEPENDENCY(example-robot-data)
ENDIF()
IF(BUILD_WITH_IPOPT)
  ADD_OPTIONAL_DEPENDENCY(ipopt)
ENDIF()
ADD_OPTIONAL_DEPENDENCY("scipy")

OPTION(BUILD_WITH_CODEGEN_SUPPORT "Build the library with the Code Generation support (required CppADCodeGen)" OFF)

OPTION(BUILD_WITH_MULTITHREADS "Build the library with the Multithreading support (required OpenMP)" OFF)
IF(BUILD_WITH_MULTITHREADS)
  # If BUILD_WITH_NTHREADS defined, use the value - otherwise detect
  IF(NOT DEFINED BUILD_WITH_NTHREADS)
    INCLUDE(ProcessorCount)
    ProcessorCount(NPROCESSOR)
    IF(NOT NPROCESSOR EQUAL 0)
      MATH(EXPR NPROCESSOR ${NPROCESSOR}/2)
      SET(BUILD_WITH_NTHREADS ${NPROCESSOR} CACHE STRING "Number of threads")
    ELSE()
      SET(BUILD_WITH_NTHREADS "4" CACHE STRING "Number of threads") # Default setting
    ENDIF()
  ENDIF()
  string(REGEX MATCH "^[0-9]+$" BUILD_WITH_NTHREADS  ${BUILD_WITH_NTHREADS})
  IF(NOT BUILD_WITH_NTHREADS MATCHES "^[0-9]+$")
    SET(BUILD_WITH_NTHREADS 4) # Default setting 4.
    MESSAGE(WARNING "The number of threads has to be an integer value, set to ${BUILD_WITH_NTHREADS}")
  ENDIF()
ENDIF()

IF(BUILD_WITH_CODEGEN_SUPPORT)
  ADD_PROJECT_DEPENDENCY(cppad 20200000.0 REQUIRED)
  ADD_PROJECT_DEPENDENCY(cppadcg 2.4.1 REQUIRED)
  ADD_DEFINITIONS(-DCROCODDYL_WITH_CODEGEN)
  #Pinocchio codegen related preproccessor defs.
  ADD_DEFINITIONS(-DPINOCCHIO_WITH_CPPADCG_SUPPORT)
  ADD_DEFINITIONS(-DPINOCCHIO_WITH_CPPAD_SUPPORT)
  #Packaging for downstream.
  SET(PACKAGE_EXTRA_MACROS "${PACKAGE_EXTRA_MACROS}\nADD_DEFINITIONS(-DCROCODDYL_WITH_CODEGEN -DPINOCCHIO_WITH_CPPAD_SUPPORT -DPINOCCHIO_WITH_CPPADCG_SUPPORT)")
  IF(NOT ${EIGEN3_VERSION} VERSION_GREATER "3.3.0")
    ADD_DEFINITIONS(-DPINOCCHIO_CPPAD_REQUIRES_MATRIX_BASE_PLUGIN)
    SET(PACKAGE_EXTRA_MACROS "${PACKAGE_EXTRA_MACROS}\nADD_DEFINITIONS(-DPINOCCHIO_CPPAD_REQUIRES_MATRIX_BASE_PLUGIN)")
  ENDIF(NOT ${EIGEN3_VERSION} VERSION_GREATER "3.3.0")
ENDIF()

# Add OpenMP
IF(BUILD_WITH_MULTITHREADS)
  FIND_PACKAGE(OpenMP)
ENDIF()
IF(OPENMP_FOUND AND BUILD_WITH_MULTITHREADS)
  IF(APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
  ELSE()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  ENDIF(APPLE)
  ADD_DEFINITIONS(-DCROCODDYL_WITH_MULTITHREADING)
  ADD_DEFINITIONS(-DCROCODDYL_WITH_NTHREADS=${BUILD_WITH_NTHREADS})
  SET(PACKAGE_EXTRA_MACROS "${PACKAGE_EXTRA_MACROS}\nADD_DEFINITIONS(-DCROCODDYL_WITH_MULTITHREADING -DCROCODDYL_WITH_NTHREADS=${BUILD_WITH_NTHREADS})")
  SET(OMP_NUM_THREADS ${BUILD_WITH_NTHREADS})
ENDIF()

# Add Ipopt
IF(BUILD_WITH_IPOPT AND IPOPT_FOUND)
  ADD_DEFINITIONS(-DCROCODDYL_WITH_IPOPT)
  LINK_DIRECTORIES(${IPOPT_LIBRARY_DIRS})
ENDIF()

SET(BOOST_REQUIRED_COMPONENTS filesystem serialization system)
SET(BOOST_BUILD_COMPONENTS unit_test_framework)
SET_BOOST_DEFAULT_OPTIONS()
EXPORT_BOOST_DEFAULT_OPTIONS()
ADD_PROJECT_DEPENDENCY(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})
FIND_PACKAGE(Boost REQUIRED COMPONENTS ${BOOST_BUILD_COMPONENTS})

IF(Boost_VERSION GREATER 107299)
  # Silence a warning about a deprecated use of boost bind by boost python
  # at least fo boost 1.73 to 1.75
  ADD_DEFINITIONS(-DBOOST_BIND_GLOBAL_PLACEHOLDERS)
ENDIF()


# Main library
SET(${PROJECT_NAME}_SOURCES
  src/core/solver-base.cpp
  src/core/solvers/box-ddp.cpp
  src/core/solvers/box-fddp.cpp
  src/core/solvers/box-qp.cpp
  src/core/solvers/ddp.cpp
  src/core/solvers/fddp.cpp
  src/core/solvers/intro.cpp
  src/core/solvers/ipopt.cpp
  src/core/solvers/ipopt/ipopt-iface.cpp
  src/core/solvers/kkt.cpp
  src/core/utils/callbacks.cpp
  src/core/utils/exception.cpp
  src/core/utils/stop-watch.cpp
  src/multibody/utils/quadruped-gaits.cpp
  )

SET(${PROJECT_NAME}_HEADERS
  include/crocoddyl/core/action-base.hpp
  include/crocoddyl/core/action-base.hxx
  include/crocoddyl/core/actions/diff-lqr.hpp
  include/crocoddyl/core/actions/diff-lqr.hxx
  include/crocoddyl/core/actions/lqr.hpp
  include/crocoddyl/core/actions/lqr.hxx
  include/crocoddyl/core/actions/unicycle.hpp
  include/crocoddyl/core/actions/unicycle.hxx
  include/crocoddyl/core/activation-base.hpp
  include/crocoddyl/core/activations/2norm-barrier.hpp
  include/crocoddyl/core/activations/quadratic-barrier.hpp
  include/crocoddyl/core/activations/quadratic-flat-exp.hpp
  include/crocoddyl/core/activations/quadratic-flat-log.hpp
  include/crocoddyl/core/activations/quadratic.hpp
  include/crocoddyl/core/activations/smooth-1norm.hpp
  include/crocoddyl/core/activations/smooth-2norm.hpp
  include/crocoddyl/core/activations/smooth-abs.hpp
  include/crocoddyl/core/activations/weighted-quadratic-barrier.hpp
  include/crocoddyl/core/activations/weighted-quadratic.hpp
  include/crocoddyl/core/actuation/actuation-squashing.hpp
  include/crocoddyl/core/actuation-base.hpp
  include/crocoddyl/core/actuation-base.hxx
  include/crocoddyl/core/actuation/squashing-base.hpp
  include/crocoddyl/core/actuation/squashing/smooth-sat.hpp
  include/crocoddyl/core/codegen/action-base.hpp
  include/crocoddyl/core/constraint-base.hpp
  include/crocoddyl/core/constraint-base.hxx
  include/crocoddyl/core/constraints/constraint-manager.hpp
  include/crocoddyl/core/constraints/constraint-manager.hxx
  include/crocoddyl/core/constraints/residual.hpp
  include/crocoddyl/core/constraints/residual.hxx
  include/crocoddyl/core/control-base.hpp
  include/crocoddyl/core/control-base.hxx
  include/crocoddyl/core/controls/poly-one.hpp
  include/crocoddyl/core/controls/poly-one.hxx
  include/crocoddyl/core/controls/poly-two-rk.hpp
  include/crocoddyl/core/controls/poly-two-rk.hxx
  include/crocoddyl/core/controls/poly-zero.hpp
  include/crocoddyl/core/controls/poly-zero.hxx
  include/crocoddyl/core/cost-base.hpp
  include/crocoddyl/core/cost-base.hxx
  include/crocoddyl/core/costs/cost-sum.hpp
  include/crocoddyl/core/costs/cost-sum.hxx
  include/crocoddyl/core/costs/residual.hpp
  include/crocoddyl/core/costs/residual.hxx
  include/crocoddyl/core/data/actuation.hpp
  include/crocoddyl/core/data-collector-base.hpp
  include/crocoddyl/core/data/joint.hpp
  include/crocoddyl/core/diff-action-base.hpp
  include/crocoddyl/core/diff-action-base.hxx
  include/crocoddyl/core/fwd.hpp
  include/crocoddyl/core/integ-action-base.hpp
  include/crocoddyl/core/integ-action-base.hxx
  include/crocoddyl/core/integrator/euler.hpp
  include/crocoddyl/core/integrator/euler.hxx
  include/crocoddyl/core/integrator/rk4.hpp
  include/crocoddyl/core/integrator/rk4.hxx
  include/crocoddyl/core/integrator/rk.hpp
  include/crocoddyl/core/integrator/rk.hxx
  include/crocoddyl/core/mathbase.hpp
  include/crocoddyl/core/numdiff/action.hpp
  include/crocoddyl/core/numdiff/action.hxx
  include/crocoddyl/core/numdiff/activation.hpp
  include/crocoddyl/core/numdiff/activation.hxx
  include/crocoddyl/core/numdiff/actuation.hpp
  include/crocoddyl/core/numdiff/actuation.hxx
  include/crocoddyl/core/numdiff/constraint.hpp
  include/crocoddyl/core/numdiff/constraint.hxx
  include/crocoddyl/core/numdiff/control.hpp
  include/crocoddyl/core/numdiff/control.hxx
  include/crocoddyl/core/numdiff/cost.hpp
  include/crocoddyl/core/numdiff/cost.hxx
  include/crocoddyl/core/numdiff/diff-action.hpp
  include/crocoddyl/core/numdiff/diff-action.hxx
  include/crocoddyl/core/numdiff/residual.hpp
  include/crocoddyl/core/numdiff/residual.hxx
  include/crocoddyl/core/numdiff/state.hpp
  include/crocoddyl/core/numdiff/state.hxx
  include/crocoddyl/core/optctrl/shooting.hpp
  include/crocoddyl/core/optctrl/shooting.hxx
  include/crocoddyl/core/residual-base.hpp
  include/crocoddyl/core/residual-base.hxx
  include/crocoddyl/core/residuals/control.hpp
  include/crocoddyl/core/residuals/control.hxx
  include/crocoddyl/core/residuals/joint-acceleration.hpp
  include/crocoddyl/core/residuals/joint-acceleration.hxx
  include/crocoddyl/core/residuals/joint-effort.hpp
  include/crocoddyl/core/residuals/joint-effort.hxx
  include/crocoddyl/core/solver-base.hpp
  include/crocoddyl/core/solvers/box-ddp.hpp
  include/crocoddyl/core/solvers/box-fddp.hpp
  include/crocoddyl/core/solvers/box-kkt.hpp
  include/crocoddyl/core/solvers/box-qp.hpp
  include/crocoddyl/core/solvers/ddp.hpp
  include/crocoddyl/core/solvers/fddp.hpp
  include/crocoddyl/core/solvers/intro.hpp
  include/crocoddyl/core/solvers/ipopt.hpp
  include/crocoddyl/core/solvers/ipopt/ipopt-iface.hpp
  include/crocoddyl/core/solvers/kkt.hpp
  include/crocoddyl/core/state-base.hpp
  include/crocoddyl/core/state-base.hxx
  include/crocoddyl/core/states/euclidean.hpp
  include/crocoddyl/core/states/euclidean.hxx
  include/crocoddyl/core/utils/callbacks.hpp
  include/crocoddyl/core/utils/deprecate.hpp
  include/crocoddyl/core/utils/exception.hpp
  include/crocoddyl/core/utils/file-io.hpp
  include/crocoddyl/core/utils/math.hpp
  include/crocoddyl/core/utils/motion-display.hpp
  include/crocoddyl/core/utils/robot-loader.hpp
  include/crocoddyl/core/utils/solution-plot.hpp
  include/crocoddyl/core/utils/stop-watch.hpp
  include/crocoddyl/core/utils/timer.hpp
  include/crocoddyl/core/utils/version.hpp
  include/crocoddyl/multibody/actions/contact-fwddyn.hpp
  include/crocoddyl/multibody/actions/contact-fwddyn.hxx
  include/crocoddyl/multibody/actions/contact-invdyn.hpp
  include/crocoddyl/multibody/actions/contact-invdyn.hxx
  include/crocoddyl/multibody/actions/free-fwddyn.hpp
  include/crocoddyl/multibody/actions/free-fwddyn.hxx
  include/crocoddyl/multibody/actions/free-invdyn.hpp
  include/crocoddyl/multibody/actions/free-invdyn.hxx
  include/crocoddyl/multibody/actions/impulse-fwddyn.hpp
  include/crocoddyl/multibody/actions/impulse-fwddyn.hxx
  include/crocoddyl/multibody/actuations/floating-base.hpp
  include/crocoddyl/multibody/actuations/full.hpp
  include/crocoddyl/multibody/actuations/multicopter-base.hpp
  include/crocoddyl/multibody/contact-base.hpp
  include/crocoddyl/multibody/contact-base.hxx
  include/crocoddyl/multibody/contacts/contact-1d.hpp
  include/crocoddyl/multibody/contacts/contact-1d.hxx
  include/crocoddyl/multibody/contacts/contact-2d.hpp
  include/crocoddyl/multibody/contacts/contact-2d.hxx
  include/crocoddyl/multibody/contacts/contact-3d.hpp
  include/crocoddyl/multibody/contacts/contact-3d.hxx
  include/crocoddyl/multibody/contacts/contact-6d.hpp
  include/crocoddyl/multibody/contacts/contact-6d.hxx
  include/crocoddyl/multibody/contacts/multiple-contacts.hpp
  include/crocoddyl/multibody/contacts/multiple-contacts.hxx
  include/crocoddyl/multibody/cop-support.hpp
  include/crocoddyl/multibody/cop-support.hxx
  include/crocoddyl/multibody/cost-base.hpp
  include/crocoddyl/multibody/data/contacts.hpp
  include/crocoddyl/multibody/data/impulses.hpp
  include/crocoddyl/multibody/data/multibody.hpp
  include/crocoddyl/multibody/force-base.hpp
  include/crocoddyl/multibody/frames-deprecated.hpp
  include/crocoddyl/multibody/frames.hpp
  include/crocoddyl/multibody/friction-cone.hpp
  include/crocoddyl/multibody/friction-cone.hxx
  include/crocoddyl/multibody/fwd.hpp
  include/crocoddyl/multibody/impulse-base.hpp
  include/crocoddyl/multibody/impulse-base.hxx
  include/crocoddyl/multibody/impulses/impulse-3d.hpp
  include/crocoddyl/multibody/impulses/impulse-3d.hxx
  include/crocoddyl/multibody/impulses/impulse-6d.hpp
  include/crocoddyl/multibody/impulses/impulse-6d.hxx
  include/crocoddyl/multibody/impulses/multiple-impulses.hpp
  include/crocoddyl/multibody/impulses/multiple-impulses.hxx
  include/crocoddyl/multibody/numdiff/contact.hpp
  include/crocoddyl/multibody/numdiff/contact.hxx
  include/crocoddyl/multibody/numdiff/cost.hpp
  include/crocoddyl/multibody/residuals/centroidal-momentum.hpp
  include/crocoddyl/multibody/residuals/centroidal-momentum.hxx
  include/crocoddyl/multibody/residuals/com-position.hpp
  include/crocoddyl/multibody/residuals/com-position.hxx
  include/crocoddyl/multibody/residuals/contact-control-gravity.hpp
  include/crocoddyl/multibody/residuals/contact-control-gravity.hxx
  include/crocoddyl/multibody/residuals/contact-cop-position.hpp
  include/crocoddyl/multibody/residuals/contact-cop-position.hxx
  include/crocoddyl/multibody/residuals/contact-force.hpp
  include/crocoddyl/multibody/residuals/contact-force.hxx
  include/crocoddyl/multibody/residuals/contact-friction-cone.hpp
  include/crocoddyl/multibody/residuals/contact-friction-cone.hxx
  include/crocoddyl/multibody/residuals/contact-wrench-cone.hpp
  include/crocoddyl/multibody/residuals/contact-wrench-cone.hxx
  include/crocoddyl/multibody/residuals/control-gravity.hpp
  include/crocoddyl/multibody/residuals/control-gravity.hxx
  include/crocoddyl/multibody/residuals/frame-placement.hpp
  include/crocoddyl/multibody/residuals/frame-placement.hxx
  include/crocoddyl/multibody/residuals/frame-rotation.hpp
  include/crocoddyl/multibody/residuals/frame-rotation.hxx
  include/crocoddyl/multibody/residuals/frame-translation.hpp
  include/crocoddyl/multibody/residuals/frame-translation.hxx
  include/crocoddyl/multibody/residuals/frame-velocity.hpp
  include/crocoddyl/multibody/residuals/frame-velocity.hxx
  include/crocoddyl/multibody/residuals/impulse-com.hpp
  include/crocoddyl/multibody/residuals/impulse-com.hxx
  include/crocoddyl/multibody/residuals/pair-collision.hpp
  include/crocoddyl/multibody/residuals/pair-collision.hxx
  include/crocoddyl/multibody/residuals/state.hpp
  include/crocoddyl/multibody/residuals/state.hxx
  include/crocoddyl/multibody/states/multibody.hpp
  include/crocoddyl/multibody/states/multibody.hxx
  include/crocoddyl/multibody/utils/quadruped-gaits.hpp
  include/crocoddyl/multibody/wrench-cone.hpp
  include/crocoddyl/multibody/wrench-cone.hxx
  )

IF(UNIX)
  # Build options
  #   We'd like to deactivate sign conversion since we frequently convert Eigen::Index<>std::size_t
  #   Otherwise, activate all warnings.
  ADD_COMPILE_OPTIONS(-Wall -Wpedantic -Wextra -Wno-sign-conversion)

  ADD_LIBRARY(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
  TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} pinocchio::pinocchio)
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}
      ${Boost_SERIALIZATION_LIBRARY})

  IF(OPENMP_FOUND)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenMP_CXX_LIBRARIES})
  ENDIF()

  IF(BUILD_WITH_IPOPT AND IPOPT_FOUND)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} ipopt)
    TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${IPOPT_INCLUDE_DIRS})
  ENDIF()

  INSTALL(TARGETS ${PROJECT_NAME} EXPORT ${TARGETS_EXPORT_NAME} DESTINATION lib)
ENDIF(UNIX)

# Build the Python interface
IF(BUILD_PYTHON_INTERFACE)
  # Include and add the bindings directory
  INCLUDE_DIRECTORIES(bindings)
  ADD_SUBDIRECTORY(bindings)
ENDIF(BUILD_PYTHON_INTERFACE)

# Build the unit tests
IF(BUILD_TESTING)
  ADD_SUBDIRECTORY(unittest)
ENDIF(BUILD_TESTING)

# Build the benchmark
IF(BUILD_BENCHMARK)
  ADD_SUBDIRECTORY(benchmark)
ENDIF(BUILD_BENCHMARK)

# Build the examples
IF(BUILD_EXAMPLES)
  IF(BUILD_PYTHON_INTERFACE)
    ADD_SUBDIRECTORY(examples)
  ELSE(BUILD_PYTHON_INTERFACE)
    MESSAGE(WARNING "Python interface is not built, hence cannot build examples.")
  ENDIF(BUILD_PYTHON_INTERFACE)
ENDIF(BUILD_EXAMPLES)

# PkgConfig packaging of the project
PKG_CONFIG_APPEND_LIBS(${PROJECT_NAME})
PKG_CONFIG_APPEND_BOOST_LIBS(${BOOST_REQUIRED_COMPONENTS})
