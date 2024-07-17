#!/bin/zsh

export PPC_HOME=${PPC_HOME:-/opt/greenxserver}
cd $PPC_HOME/ragagent/python
source .venv/bin/activate
python3 main.py "$1" "$2"
deactivate
