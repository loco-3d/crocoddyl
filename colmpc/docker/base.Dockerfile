FROM ubuntu:22.04

RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -qqy --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    coinor-libipopt-dev \
    git \
    libassimp-dev \
    libboost-all-dev \
    libeigen3-dev \
    liboctomap-dev \
    libqhull-dev \
    libtinyxml-dev \
    liburdfdom-dev \
    python-is-python3 \
    python3-dev \
    python3-numpy \
    python3-scipy

WORKDIR /src
ADD docker/base.sh .
RUN ./base.sh /usr/local \
    && rm -rf *
