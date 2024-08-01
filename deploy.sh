#!/bin/zsh

cp -f ragagent.properties "$PPC_HOME"/config/properties/
ls -l "$PPC_HOME"/config/properties/ragagent.properties

cp -f virtualenv.sh run.sh "$PPC_HOME"/ragagent/
chmod +x "$PPC_HOME"/ragagent/*.sh

cp -f src/model.py src/core.py src/main.py src/requirements.txt "$PPC_HOME"/ragagent/python
ls -l "$PPC_HOME"/ragagent/python

"$PPC_HOME"/ragagent/virtualenv.sh
