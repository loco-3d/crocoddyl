#!/bin/bash
set -xe

# Exit if not in format testing mode
if [ -z $CHECK_CLANG_FORMAT ]; then exit 0; fi

sudo pip install flake8 isort yapf==0.28.0
sudo apt install -y -qq clang-format-6.0
cd $TRAVIS_BUILD_DIR && wget https://raw.githubusercontent.com/Gepetto/linters/master/.clang-format
yapf -ri . && flake8 . && clang-format-6.0 -i $(find . -path ./cmake -prune -o -iregex '.*\.\(h\|c\|hh\|cc\|hpp\|cpp\|hxx\|cxx\)$' -print)
git diff --ignore-submodules
exit $(git diff --ignore-submodules | wc -l)
