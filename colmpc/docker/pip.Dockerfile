FROM ubuntu:22.04

RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -qqy --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    python-is-python3 \
    python3-dev \
    python3-pip \
    python3-venv

WORKDIR /src
ADD pip.sh .
RUN ./pip.sh
