#!/bin/bash -eux

rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install eigenpy[build]==3.5.1 cmeel-assimp cmeel-octomap cmeel-qhull
pip install git+https://github.com/cmake-wheel/hpp-fcl.git@cmeel-devel
pip install pin[build]==2.7.0 example-robot-data[build]==4.1.0 crocoddyl[build]==2.1.0 cmeel-mim-solvers[build]==0.0.4

CMAKE_PREFIX_PATH="$(cmeel cmake)"
export CMAKE_PREFIX_PATH

git clone --branch main --recursive https://github.com/agimus-project/colmpc
cmake -B build -S colmpc -DCMAKE_INSTALL_PREFIX="$(cmeel cmake)"
cmake --build build
cmake --build build -t install
