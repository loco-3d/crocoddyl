#!/bin/bash
set -xe

# Exit if in format testing mode
if [ $CHECK_CLANG_FORMAT ]; then exit 0; fi

#sudo apt upgrade -y -qq
sudo apt install -y -qq libeigen3-dev doxygen robotpkg-py27-eigenpy robotpkg-py27-pinocchio robotpkg-py27-example-robot-data python-scipy robotpkg-py27-quadprog