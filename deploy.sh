#!/bin/zsh

cp -f ragagent.properties "$PPC_HOME"/config/properties/
ls -l "$PPC_HOME"/config/properties/ragagent.properties

pushd src || exit
cp -f model.py core.py main.py main.sh ../requirements.txt "$PPC_HOME"/ragagent/python
ls -l "$PPC_HOME"/ragagent/python
popd || exit

pushd "$PPC_HOME"/ragagent/python || exit
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

deactivate

popd || exit