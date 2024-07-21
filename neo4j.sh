#!/bin/zsh

NEO4J_VERSION=5.21.0

NEO4J_CONTAINER_ID=$(docker ps -aq --filter ancestor=neo4j:"$NEO4J_VERSION")

NEO4J_HOME=/Users/oleg/Projects/PPC/home/ragagent/neo4j

if [ -n "$NEO4J_CONTAINER_ID" ]
then
  # Remove existing Neo4j container
  docker rm -f "$NEO4J_CONTAINER_ID"
  # Remove data
  rm -rf $NEO4J_HOME/data/*
  rm -rf $NEO4J_HOME/logs/*
fi

# Run a new container
docker run -d \
    --restart always \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/pe0p1eP0wer \
    --volume=$NEO4J_HOME/data:/data \
    --volume=$NEO4J_HOME/logs:/logs \
    neo4j:"$NEO4J_VERSION"
