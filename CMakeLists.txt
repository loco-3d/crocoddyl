cmake_minimum_required(VERSION 3.10)

# Set up project properties
set(PROJECT_NAMESPACE loco-3d)
set(PROJECT_NAME crocoddyl)
set(PROJECT_DESCRIPTION
    "Contact RObot COntrol by Differential DYnamic programming Library (Crocoddyl)"
)
set(PROJECT_URL https://github.com/${PROJECT_NAMESPACE}/${PROJECT_NAME})

option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(EXISTS "${JRL_CMAKE_MODULES}/base.cmake")
  message(STATUS "JRL cmakemodules found in 'cmake/' git submodule")
else()
  find_package(jrl-cmakemodules QUIET CONFIG)
  if(jrl-cmakemodules_FOUND)
    get_property(
      JRL_CMAKE_MODULES
      TARGET jrl-cmakemodules::jrl-cmakemodules
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}")
  elseif(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    message(
      FATAL_ERROR
        "\nCan't find jrl-cmakemodules. Please either:\n"
        "  - use git submodule: 'git submodule update --init'\n"
        "  - or install https://github.com/jrl-umi3218/jrl-cmakemodules\n"
        "  - or upgrade your CMake version to >= 3.14 to allow automatic fetching\n"
    )
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare(
      "jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

# Include important cmake modules
include("${JRL_CMAKE_MODULES}/base.cmake")
include("${JRL_CMAKE_MODULES}/boost.cmake")
include("${JRL_CMAKE_MODULES}/apple.cmake")
include("${JRL_CMAKE_MODULES}/stubs.cmake")
include("${JRL_CMAKE_MODULES}/python-helpers.cmake")

# Print initial message
message(STATUS "${PROJECT_DESCRIPTION}, version ${PROJECT_VERSION}")
message(
  STATUS "Copyright (C) 2018-2024 CNRS-LAAS, University of Edinburgh, INRIA")
message(STATUS "                        Heriot-Watt University")
message(STATUS "All rights reserved.")
message(STATUS "Released under the BSD 3-Clause License.")

# Set a default build type to 'Release' if none was specified
set_default_cmake_build_type("Release")

set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_USE_TEMPLATE_CSS YES)

set(CXX_DISABLE_WERROR False)
set(PROJECT_USE_CMAKE_EXPORT True)

# Create different building options
option(ENABLE_VECTORIZATION
       "Enable vectorization and further processor-related optimizations" OFF)
option(BUILD_PYTHON_INTERFACE "Build the python binding" ON)
option(BUILD_BENCHMARK "Build the benchmark" ON)
option(BUILD_EXAMPLES "Build the examples" ON)
option(BUILD_WITH_IPOPT "Build the IPOPT wrapper." ON)
option(GENERATE_PYTHON_STUBS
       "Generate the Python stubs associated to the Python library" OFF)

if(ENABLE_VECTORIZATION)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

# If needed, fix CMake policy for APPLE systems
apply_default_apple_configuration()

check_minimal_cxx_standard(14 ENFORCE)

if(APPLE) # Use the handmade approach
  set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/find-external/OpenMP
                        ${CMAKE_MODULE_PATH})
elseif(UNIX)
  if(${CMAKE_VERSION} VERSION_GREATER "3.25.0" OR ${CMAKE_VERSION}
                                                  VERSION_EQUAL "3.25.0")
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/find-external/OpenMP
                          ${CMAKE_MODULE_PATH})
  endif()
endif()

# Add the different required and optional dependencies
if(BUILD_PYTHON_INTERFACE)
  add_project_dependency(eigenpy 3.1.0 REQUIRED PKG_CONFIG_REQUIRES
                         "eigenpy >= 3.1.0")
endif(BUILD_PYTHON_INTERFACE)
add_project_dependency(pinocchio 3.4.0 REQUIRED PKG_CONFIG_REQUIRES
                       "pinocchio >= 3.4.0")
if(BUILD_EXAMPLES
   OR BUILD_TESTING
   OR BUILD_BENCHMARK)
  add_project_dependency(example-robot-data 4.0.9 REQUIRED PKG_CONFIG_REQUIRES
                         "example-robot-data >= 4.0.9")
else()
  add_optional_dependency(example-robot-data)
endif()
if(BUILD_WITH_IPOPT)
  add_optional_dependency(ipopt)
endif()
add_optional_dependency("scipy")

option(
  BUILD_WITH_CODEGEN_SUPPORT
  "Build the library with the Code Generation support (required CppADCodeGen)"
  OFF)

option(BUILD_WITH_MULTITHREADS
       "Build the library with the Multithreading support (required OpenMP)"
       OFF)
if(BUILD_WITH_MULTITHREADS)
  # If BUILD_WITH_NTHREADS defined, use the value - otherwise detect
  if(NOT DEFINED BUILD_WITH_NTHREADS)
    include(ProcessorCount)
    ProcessorCount(NPROCESSOR)
    if(NOT NPROCESSOR EQUAL 0)
      math(EXPR NPROCESSOR ${NPROCESSOR}/2)
      set(BUILD_WITH_NTHREADS
          ${NPROCESSOR}
          CACHE STRING "Number of threads")
    else()
      set(BUILD_WITH_NTHREADS
          "4"
          CACHE STRING "Number of threads") # Default setting
    endif()
  endif()
  string(REGEX MATCH "^[0-9]+$" BUILD_WITH_NTHREADS ${BUILD_WITH_NTHREADS})
  if(NOT BUILD_WITH_NTHREADS MATCHES "^[0-9]+$")
    set(BUILD_WITH_NTHREADS 4) # Default setting 4.
    message(
      WARNING
        "The number of threads has to be an integer value, set to ${BUILD_WITH_NTHREADS}"
    )
  endif()
endif()

if(BUILD_WITH_CODEGEN_SUPPORT)
  set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/find-external/CppAD/"
                        ${CMAKE_MODULE_PATH})
  add_project_dependency(cppad 20200000.0 REQUIRED FIND_EXTERNAL cppad)
  add_project_dependency(cppadcg 2.4.1 REQUIRED)
  add_definitions(-DCROCODDYL_WITH_CODEGEN)
  # Pinocchio codegen related preproccessor defs.
  add_definitions(-DPINOCCHIO_WITH_CPPADCG_SUPPORT)
  add_definitions(-DPINOCCHIO_WITH_CPPAD_SUPPORT)
  # Packaging for downstream.
  set(PACKAGE_EXTRA_MACROS
      "${PACKAGE_EXTRA_MACROS}\nADD_DEFINITIONS(-DCROCODDYL_WITH_CODEGEN -DPINOCCHIO_WITH_CPPAD_SUPPORT -DPINOCCHIO_WITH_CPPADCG_SUPPORT)"
  )
  if(NOT ${EIGEN3_VERSION} VERSION_GREATER "3.3.0")
    add_definitions(-DPINOCCHIO_CPPAD_REQUIRES_MATRIX_BASE_PLUGIN)
    set(PACKAGE_EXTRA_MACROS
        "${PACKAGE_EXTRA_MACROS}\nADD_DEFINITIONS(-DPINOCCHIO_CPPAD_REQUIRES_MATRIX_BASE_PLUGIN)"
    )
  endif()
endif()

# Add OpenMP
if(BUILD_WITH_MULTITHREADS)
  find_package(OpenMP REQUIRED COMPONENTS CXX)

  add_definitions(-DCROCODDYL_WITH_MULTITHREADING)
  add_definitions(-DCROCODDYL_WITH_NTHREADS=${BUILD_WITH_NTHREADS})
  set(PACKAGE_EXTRA_MACROS
      "${PACKAGE_EXTRA_MACROS}\nADD_DEFINITIONS(-DCROCODDYL_WITH_MULTITHREADING -DCROCODDYL_WITH_NTHREADS=${BUILD_WITH_NTHREADS})"
  )
  set(OMP_NUM_THREADS ${BUILD_WITH_NTHREADS})
endif()

# Add Ipopt
if(BUILD_WITH_IPOPT AND IPOPT_FOUND)
  add_definitions(-DCROCODDYL_WITH_IPOPT)
  link_directories(${IPOPT_LIBRARY_DIRS})
endif()

set(BOOST_REQUIRED_COMPONENTS filesystem serialization system)
set(BOOST_BUILD_COMPONENTS unit_test_framework)
set_boost_default_options()
export_boost_default_options()
add_project_dependency(Boost 1.65 REQUIRED COMPONENTS
                       ${BOOST_REQUIRED_COMPONENTS})
find_package(Boost 1.65 REQUIRED COMPONENTS ${BOOST_BUILD_COMPONENTS})

if(Boost_VERSION GREATER 107299)
  # Silence a warning about a deprecated use of boost bind by boost python at
  # least fo boost 1.73 to 1.75
  add_definitions(-DBOOST_BIND_GLOBAL_PLACEHOLDERS)
endif()

# Include header and source files
file(GLOB_RECURSE ${PROJECT_NAME}_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE ${PROJECT_NAME}_HEADERS ${PROJECT_SOURCE_DIR}/include/*.hpp
     ${PROJECT_SOURCE_DIR}/include/*.hxx)
# Remove header and source files that we want to build
if(NOT BUILD_WITH_IPOPT OR (NOT IPOPT_FOUND))
  list(REMOVE_ITEM ${PROJECT_NAME}_HEADERS
       ${PROJECT_SOURCE_DIR}/include/core/solvers/ipopt/ipopt-iface.cpp
       ${PROJECT_SOURCE_DIR}/include/core/solvers/ipopt.cpp)
endif()
if(NOT BUILD_WITH_IPOPT OR (NOT IPOPT_FOUND))
  list(REMOVE_ITEM ${PROJECT_NAME}_SOURCES
       ${PROJECT_SOURCE_DIR}/src/core/solvers/ipopt/ipopt-iface.cpp
       ${PROJECT_SOURCE_DIR}/src/core/solvers/ipopt.cpp)
endif()

if(UNIX)
  # Build options We'd like to deactivate sign conversion since we frequently
  # convert Eigen::Index<>std::size_t Otherwise, activate all warnings.
  add_compile_options(-Wall -Wpedantic -Wextra -Wno-sign-conversion)

  add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES}
                                     ${${PROJECT_NAME}_HEADERS})
  target_include_directories(${PROJECT_NAME}
                             PUBLIC $<INSTALL_INTERFACE:include>)
  target_link_libraries(${PROJECT_NAME} pinocchio::pinocchio)
  target_link_libraries(${PROJECT_NAME} Boost::filesystem Boost::system
                        Boost::serialization)
  target_compile_definitions(
    ${PROJECT_NAME} PUBLIC PINOCCHIO_ENABLE_COMPATIBILITY_WITH_VERSION_2)
  set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

  if(BUILD_WITH_MULTITHREADS)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
  endif()

  if(BUILD_WITH_IPOPT AND IPOPT_FOUND)
    target_link_libraries(${PROJECT_NAME} ipopt)
    target_include_directories(${PROJECT_NAME} PUBLIC ${IPOPT_INCLUDE_DIRS})
  endif()

  install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${TARGETS_EXPORT_NAME}
    DESTINATION lib)
endif()

# Build the Python interface
if(BUILD_PYTHON_INTERFACE)
  # Include and add the bindings directory
  include_directories(bindings)
  add_subdirectory(bindings)
endif()

# Build the unit tests
if(BUILD_TESTING)
  add_subdirectory(unittest)
endif()

# Build the benchmark
if(BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

# Build the examples
if(BUILD_EXAMPLES)
  if(BUILD_PYTHON_INTERFACE)
    add_subdirectory(examples)
  else()
    message(
      WARNING "Python interface is not built, hence cannot build examples.")
  endif()
endif()

# PkgConfig packaging of the project
pkg_config_append_libs(${PROJECT_NAME})
foreach(boostlib ${BOOST_REQUIRED_COMPONENTS})
  pkg_config_append_libs("boost_${boostlib}")
endforeach()
