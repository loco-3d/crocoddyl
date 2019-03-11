FROM gepgitlab.laas.fr:4567/gepetto/buildfarm/robotpkg-jrl:16.04

RUN apt-get update -qqy \
 && apt-get install -qqy \
    flake8 \
    isort \
    python-pip \
    robotpkg-py27-pinocchio \
    robotpkg-talos-data \
 && rm -rf /var/lib/apt/lists/* \
 && pip install yapf
