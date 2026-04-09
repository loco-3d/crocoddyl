ARG COLMPC_COMMIT_HASH=main
ARG BASE_IMAGE=ghcr.io/agimus-project/colmpc:base-latest
ARG MAKE_JOBS=4

FROM $BASE_IMAGE

RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -qqy --no-install-recommends python3-pip \
    && pip install \
        meshcat \
        numdifftools

RUN git clone https://github.com/agimus-project/colmpc.git \
    && cd colmpc \
    && git checkout $COLMPC_COMMIT_HASH \
    && git submodule update --init --recursive \
    && cmake -B "build" \
        -DBUILD_TESTING=ON \
        -DBUILD_WITH_COLLISION_SUPPORT=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="/usr/local" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCOAL_BACKWARD_COMPATIBILITY_WITH_HPP_FCL=ON \
        -DCOAL_HAS_QHULL=ON \
        -DBUILD_PYTHON_INTERFACE=ON \
        -Wno-dev \
    && cmake --build "build" -j $MAKE_JOBS \
    && cmake --build "build" -t install
