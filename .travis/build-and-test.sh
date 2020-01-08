#!/bin/bash
set -xe

# Exit if in format testing mode
if [ $CHECK_CLANG_FORMAT ]; then exit 0; fi

mkdir _build ; cd _build
if [ -z "$CMAKE_CXX_STANDARD" ]
then
  cmake .. -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DCMAKE_CXX_STANDARD=$CMAKE_CXX_STANDARD
else
  cmake .. -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DCMAKE_CXX_STANDARD=$CMAKE_CXX_STANDARD
fi

make -j1
make test

# Run the examples in Release mode
if [ "$CMAKE_BUILD_TYPE" == "Release" ]
then
  # Examples
  make -s examples-double_pendulum
  make -s examples-quadrotor
  make -s examples-quadrotor_ubound
  make -s examples-arm_manipulation
  make -s examples-bipedal_walk
  make -s examples-bipedal_walk_ubound
  make -s examples-quadrupedal_gaits
  make -s examples-quadrupedal_walk_ubound
  make -s examples-humanoid_manipulation
  make -s examples-humanoid_manipulation_ubound
fi
