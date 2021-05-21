#!/bin/bash
set -xe

# Exit if in format testing mode
if [ $CHECK_CLANG_FORMAT ]; then exit 0; fi

mkdir _build ; cd _build
if [ $DIST = focal ]
then
  cmake .. -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DPYTHON_EXECUTABLE=$(which python3)
else
  cmake .. -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
fi

make -j1
make test
