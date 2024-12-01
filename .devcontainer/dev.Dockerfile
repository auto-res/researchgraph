FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    build-essential \
    locales && \
    rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN curl -LsSf https://astral.sh/uv/0.5.5/install.sh | bash || exit 1

ENV PATH="/root/.uv/bin:$PATH"

RUN bash -lc 'uv python install 3.10'
