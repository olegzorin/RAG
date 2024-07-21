#!/bin/zsh

pushd src || exit
cp -f model.py core.py main.py ../requirements.txt "$PPC_SERVER"/ragagent/src/main/python
ls -l "$PPC_SERVER"/ragagent/src/main/python
popd || exit