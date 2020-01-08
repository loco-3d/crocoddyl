#!/bin/bash
set -xe

# Exit if in format testing mode
if [ $CHECK_CLANG_FORMAT ]; then exit 0; fi

mkdir _build ; cd _build
if [ -z $CMAKE_CXX_STANDARD ]
then
  cmake .. -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
else
  cmake .. -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DCMAKE_CXX_STANDARD=$CMAKE_CXX_STANDARD
fi

make -j1
make test
