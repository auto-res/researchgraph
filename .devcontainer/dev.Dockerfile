FROM ubuntu:22.04

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    locales && \
    rm -rf /var/lib/apt/lists/*

# ロケールの設定
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# uvのインストール
RUN curl -LsSf https://astral.sh/uv/install.sh | bash || exit 1

# uvのPATH設定
ENV PATH="/root/.uv/bin:$PATH"

RUN bash -lc 'uv python install 3.10'
