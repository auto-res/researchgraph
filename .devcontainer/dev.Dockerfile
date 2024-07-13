FROM python:3.9.13-slim

USER vscode

ARG PYTHON_VIRTUALENV_HOME=/home/vscode/researchchain-py-env \
    POETRY_VERSION=1.3.2

ENV POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=true

RUN python3 -m pip install --user pipx && \
    python3 -m pipx ensurepath && \
    pipx install poetry==${POETRY_VERSION}


RUN python3 -m venv ${PYTHON_VIRTUALENV_HOME} && \
    $PYTHON_VIRTUALENV_HOME/bin/pip install --upgrade pip

ENV PATH="$PYTHON_VIRTUALENV_HOME/bin:$PATH" \
    VIRTUAL_ENV=$PYTHON_VIRTUALENV_HOME

RUN poetry completions bash >> /home/vscode/.bash_completion && \
    echo "export PATH=$PYTHON_VIRTUALENV_HOME/bin:$PATH" >> ~/.bashrc


WORKDIR /workspaces/researchchain

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-interaction --no-ansi --with dev,test,docs
