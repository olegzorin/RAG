#!/bin/zsh

CURR_DIR=$(pwd)

pushd "$PPC_SERVER"/ragagent/src/main/python || exit
cp -f core.py main.py requirements.txt "$CURR_DIR"/src
popd || exit