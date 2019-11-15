#!/bin/bash
set -xe

# Exit if in format testing mode
if [ $CHECK_CLANG_FORMAT ]; then exit 0; fi

mkdir _build ; cd _build ; cmake .. -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
make -j2
make test
