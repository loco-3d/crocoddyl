cmake_minimum_required(VERSION 3.22)

# Project properties
set(PROJECT_NAME colmpc)
set(PROJECT_DESCRIPTION "Collision avoidance for MPC")
set(PROJECT_URL https://github.com/agimus-project/${PROJECT_NAME})

# Project options
option(SUFFIX_SO_VERSION "Suffix library name with its version" ON)
option(
    GENERATE_PYTHON_STUBS
    "Generate the Python stubs associated to the Python library"
    OFF
)

# Project configuration
set(PROJECT_USE_CMAKE_EXPORT TRUE)

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
            PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        )
        message(
            STATUS
            "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}"
        )
    else()
        message(STATUS "JRL cmakemodules not found. Let's fetch it.")
        include(FetchContent)
        FetchContent_Declare(
            "jrl-cmakemodules"
            GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git"
        )
        FetchContent_MakeAvailable("jrl-cmakemodules")
        FetchContent_GetProperties(
            "jrl-cmakemodules"
            SOURCE_DIR JRL_CMAKE_MODULES
        )
    endif()
endif()

# JRL-cmakemodule setup
include("${JRL_CMAKE_MODULES}/base.cmake")
include("${JRL_CMAKE_MODULES}/boost.cmake")
include("${JRL_CMAKE_MODULES}/stubs.cmake")
include("${JRL_CMAKE_MODULES}/apple.cmake")

# Project definition
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

add_compile_options(-Wno-error)

apply_default_apple_configuration()

# Project dependencies
if(BUILD_PYTHON_INTERFACE)
    add_project_dependency(eigenpy 3.0.0 REQUIRED)
    set(PY_NAME "${PROJECT_NAME}_pywrap")
    set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB})
endif()

add_project_dependency(OpenMP REQUIRED)
add_project_dependency(crocoddyl REQUIRED)

# Main Library
if(BUILD_STANDALONE_PYTHON_INTERFACE)
    add_project_dependency(${PROJECT_NAME} REQUIRED CONFIG)
else()
    set(${PROJECT_NAME}_HEADERS
        include/${PROJECT_NAME}/fwd.hpp
        include/${PROJECT_NAME}/residual-distance-collision.hpp
        include/${PROJECT_NAME}/residual-distance-collision.hxx
        include/${PROJECT_NAME}/residual-velocity-avoidance.hpp
        include/${PROJECT_NAME}/residual-velocity-avoidance.hxx
        include/${PROJECT_NAME}/activation-distance-quadratic.hpp
        include/${PROJECT_NAME}/quadratic-exp.hpp
    )

    add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_HEADERS})
    add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
    set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
    target_include_directories(
        ${PROJECT_NAME}
        PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(${PROJECT_NAME} PUBLIC crocoddyl::crocoddyl)
    if(UNIX)
        get_relative_rpath(${CMAKE_INSTALL_LIBDIR} ${PROJECT_NAME}_INSTALL_RPATH)
        set_target_properties(
            ${PROJECT_NAME}
            PROPERTIES INSTALL_RPATH "${${PROJECT_NAME}_INSTALL_RPATH}"
        )
    endif()

    if(SUFFIX_SO_VERSION)
        set_target_properties(
            ${PROJECT_NAME}
            PROPERTIES SOVERSION ${PROJECT_VERSION}
        )
    endif()
endif()

# Python Bindings
if(BUILD_PYTHON_INTERFACE)
    add_subdirectory(python)
endif()

# # Unit tests
if(BUILD_TESTING)
    # find_package(Boost REQUIRED COMPONENTS  unit_test_framework)
    add_subdirectory(tests)
endif()

# Installation
if(NOT BUILD_STANDALONE_PYTHON_INTERFACE)
    install(
        TARGETS ${PROJECT_NAME}
        EXPORT ${TARGETS_EXPORT_NAME}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()
