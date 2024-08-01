#!/bin/zsh

# We use a docker container to run Neo4J
#
# This script stops and deletes existing container and removes
# all existing data and logs, if any.
# Finally, the script runs a new container.
#
# Before running this script, make sure the docker
# is installed and running on the server.

NEO4J_HOME=${PPC_HOME:-/opt/greenxserver}/ragagent/neo4j

NEO4J_VERSION=5.21.0

# Delete existing Neo4j container(s) if any
for id in $(docker ps --all --quiet --filter ancestor=neo4j:"$NEO4J_VERSION")
do
  docker rm --volumes --force "$id"
done

# Pull the Neo4j docker image to the local repository.
# This command does nothing if such an image already exists in the local repo.
docker pull neo4j:"$NEO4J_VERSION"

# Specify any password for a new Neo4j instance. It must coincide with the value
# of ppc.ragagent.neo4j.password from ragagent.properties
NEO4J_PASSWORD=pr0baPera

# Clear old data and logs and run a new container in background
rm -rf "$NEO4J_HOME"/data/*
rm -rf "$NEO4J_HOME"/logs/*

docker run --detach \
    --restart always \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/"$NEO4J_PASSWORD" \
    --volume="$NEO4J_HOME"/data:/data \
    --volume="$NEO4J_HOME"/logs:/logs \
    neo4j:"$NEO4J_VERSION"
