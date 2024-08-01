#!/bin/zsh

# This script creates a virtual environment (venv) if it does not exist,
# activates it, and installs the Python packages listed in the requirement.txt
# into it.
#
#  Before running the script, make sure the /opt/greenxserver/ragagent/python/
#  directory exists and the requirements.txt file is copied to that directory.
#
#  This script should be run during initial setup and after any changes to
#  the requirements.txt file.
#

pushd "${PPC_HOME:-/opt/greenxserver}"/ragagent/python || exit

if [ ! -d ".venv" ]
then
  # Create a virtual environment
  ${PPC_PYTHON:-python3.11} -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

deactivate

popd || exit
