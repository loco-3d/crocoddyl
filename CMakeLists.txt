CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Check if the cmake (git submodule) have been initialized
IF(NOT EXISTS "${PROJECT_SOURCE_DIR}/cmake/base.cmake")
  MESSAGE(FATAL_ERROR "\nPlease run the following command first:\ngit submodule update --init\n")
ENDIF()

# Include important cmake modules
INCLUDE(cmake/base.cmake)
INCLUDE(cmake/test.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/python.cmake)

# Set up project properties
SET(PROJECT_NAMESPACE loco-3d)
SET(PROJECT_NAME crocoddyl)
SET(PROJECT_DESCRIPTION "Contact RObot COntrol by Differential DYnamic programming Library (Crocoddyl)")
SET(PROJECT_URL https://gepgitlab.laas.fr/${PROJECT_NAMESPACE}/${PROJECT_NAME})


# Print initial message
MESSAGE("${PROJECT_DESCRIPTION}, version ${PROJECT_VERSION}")
MESSAGE("Copyright (C) 2018-2019 CNRS-LAAS")
MESSAGE("All rights reserved.")
MESSAGE("Released under the BSD 3-Clause License.")


# Set a default build type to 'Release' if none was specified
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  MESSAGE(STATUS "Setting build type to 'Release' as none was specified.")
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF()

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)

# Create different building options
OPTION(ENABLE_VECTORIZATION "Enable vectorization and futhers processor-related optimizations" ON)
OPTION(BUILD_PYTHON_INTERFACE "Build the python binding" OFF)
OPTION(BUILD_UNIT_TESTS "Build the unitary tests" ON)
OPTION(BUILD_BENCHMARK "Build the benchmark" OFF)


SET(CMAKE_CXX_FLAGS "-std=c++11")
IF(ENABLE_VECTORIZATION)
  SET(CMAKE_CXX_FLAGS "-std=c++11 -march=native -mavx")
ENDIF()

SETUP_PROJECT()

# Add the different required and optional dependencies
ADD_REQUIRED_DEPENDENCY("eigen3 >= 3.0.5")
ADD_REQUIRED_DEPENDENCY("eigenpy")
ADD_REQUIRED_DEPENDENCY("pinocchio >= 2.1.0")
ADD_OPTIONAL_DEPENDENCY("multicontact-api >= 1.1.0")
ADD_OPTIONAL_DEPENDENCY("quadprog")
ADD_OPTIONAL_DEPENDENCY("scipy")

FINDPYTHON()


# Include and add the main directory
ADD_SUBDIRECTORY(src)


# Build the Python interface
IF(BUILD_PYTHON_INTERFACE)
  SET(BOOST_COMPONENTS ${BOOST_COMPONENTS} python)
  SEARCH_FOR_BOOST()
  INCLUDE_DIRECTORIES(SYSTEM ${PYTHON_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

  # Include and add the bindings directory
  INCLUDE_DIRECTORIES(bindings)
  ADD_SUBDIRECTORY(bindings)
ENDIF(BUILD_PYTHON_INTERFACE)

# Build the unit tests
IF(BUILD_UNIT_TESTS)
  ADD_SUBDIRECTORY(unittest)
ENDIF(BUILD_UNIT_TESTS)


SETUP_PROJECT_FINALIZE()

