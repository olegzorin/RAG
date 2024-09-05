#!/bin/zsh

cd "${PPC_HOME:-/opt/greenxserver}"/ragagent/neo4j || exit

NEO4J_IMAGE=neo4j_apoc:5.21.0

docker build -t $NEO4J_IMAGE .