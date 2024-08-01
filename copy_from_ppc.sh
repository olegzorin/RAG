#!/bin/zsh

CURR_DIR=$(pwd)

pushd "$PPC_SERVER"/ragagent/src/main/python || exit
cp -f model.py core.py main.py requirements.txt "$CURR_DIR"/src
popd || exit