FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="${PATH}:/root/.local/bin"
RUN poetry config virtualenvs.in-project true

WORKDIR /app
COPY pyproject.toml poetry.lock /app/

#RUN poetry install 

ENV PATH="/app/.venv/bin:$PATH"
