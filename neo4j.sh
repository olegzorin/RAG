#!/bin/zsh

docker rm -f "$(docker ps -q)"

export NEO4J_HOME=/Users/oleg/Projects/PPC/home/ragagent/neo4j

rm -rf $NEO4J_HOME/data/*
rm -rf $NEO4J_HOME/logs/*

docker run -d \
    --restart always \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/pe0p1eP0wer \
    --volume=$NEO4J_HOME/data:/data \
    --volume=$NEO4J_HOME/logs:/logs \
    neo4j:5.21.0
