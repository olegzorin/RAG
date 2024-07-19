#!/bin/zsh

cp -f ragagent.properties "$PPC_HOME"/config/properties/
ls -l "$PPC_HOME"/config/properties/ragagent.properties

pushd src || exit
cp -f model.py core.py main.py main.sh ../requirements.txt "$PPC_HOME"/ragagent/python
ls -l "$PPC_HOME"/ragagent/python
popd || exit