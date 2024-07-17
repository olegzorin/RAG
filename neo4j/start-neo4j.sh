#!/bin/zsh

docker rm -f "$(docker ps -q)"

rm -rf /Users/oleg/Projects/PPC-RAG/neo4j/data/*
rm -rf /Users/oleg/Projects/PPC-RAG/neo4j/logs/*

docker run -d \
    --restart always \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/pe0p1eP0wer \
    --volume=/Users/oleg/Projects/PPC-RAG/neo4j/data:/data \
    --volume=/Users/oleg/Projects/PPC-RAG/neo4j/logs:/logs \
    neo4j:5.21.0

