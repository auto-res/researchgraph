# 使用するUbuntuのバージョンを指定
FROM ubuntu:22.04

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl

RUN ln -s /usr/bin/python3 /usr/bin/python

# Poetryのインストール
RUN curl -sSL https://install.python-poetry.org | python3 -

# 環境変数の設定
ENV PATH="${PATH}:/root/.local/bin"

# 作業ディレクトリの設定
WORKDIR /app

# pyproject.toml と poetry.lock ファイルをコピー
COPY pyproject.toml /app/
COPY poetry.lock /app/

# Poetryを使って依存関係をインストール
RUN poetry install

# Poetryの仮想環境パスをPATHに追加
ENV PATH="/workspaces/llmlink/.venv/bin:$PATH"
