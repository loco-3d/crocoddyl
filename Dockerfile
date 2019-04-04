FROM eur0c.laas.fr:5000/gepetto/buildfarm/robotpkg:16.04

RUN apt-get update -qqy \
 && apt-get install -qqy \
    cython \
    flake8 \
    isort \
    python-pip \
    python-scipy \
    robotpkg-example-robot-data \
    robotpkg-py27-pinocchio \
    robotpkg-py27-multicontact-api \
 && rm -rf /var/lib/apt/lists/* \
 && pip install \
    quadprog \
    yapf
