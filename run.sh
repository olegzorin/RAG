#!/bin/zsh

export PPC_HOME=${PPC_HOME:-/opt/greenxserver}

cd "${PPC_HOME}"/ragagent/python || exit 1

source .venv/bin/activate

python main.py "$1"

deactivate
