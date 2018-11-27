FROM gepgitlab.laas.fr:4567/gepetto/buildfarm/robotpkg-jrl:16.04

RUN apt-get update -qqy \
 && apt-get install -qqy \
    robotpkg-py27-pinocchio \
    robotpkg-talos-data \
 && rm -rf /var/lib/apt/lists/*
