FROM python:3.9.13-slim

#ARG PYTHON_VIRTUALENV_HOME=/home/vscode/researchchain-py-env

#ENV VIRTUAL_ENV=$PYTHON_VIRTUALENV_HOME
#ENV PATH="$PYTHON_VIRTUALENV_HOME/bin:$PATH"

#RUN python3 -m venv ${PYTHON_VIRTUALENV_HOME} && \
#    $PYTHON_VIRTUALENV_HOME/bin/pip install --upgrade pip

WORKDIR /workspaces/researchchain

#COPY pyproject.toml poetry.lock ./

RUN pip install openai
