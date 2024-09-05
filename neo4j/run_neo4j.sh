#!/bin/zsh

# We use a docker container to run Neo4J
#
# This script stops and deletes existing container and removes
# all existing data and logs, if any.
# Finally, the script runs a new container.
#
# Before running this script, make sure the docker
# is installed and running on the server.

NEO4J_IMAGE=neo4j_apoc:5.21.0

# Delete existing Neo4j container(s) if any
for id in $(docker ps --all --quiet --filter ancestor="$NEO4J_IMAGE")
do
  docker rm --volumes --force "$id"
done

# Specify any password for a new Neo4j instance. It must coincide with the value
# of ppc.ragagent.neo4j.password from ragagent.properties
NEO4J_PASSWORD=pr0baPera

NEO4J_LOGS=${PPC_HOME:-/opt/greenxserver}/ragagent/neo4j/logs

docker run --detach \
    --restart always \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/"$NEO4J_PASSWORD" \
    --volume="$NEO4J_LOGS":/logs \
    "$NEO4J_IMAGE"
