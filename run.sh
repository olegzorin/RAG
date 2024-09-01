#!/bin/zsh

export PPC_HOME=${PPC_HOME:-/opt/greenxserver}

source .venv/bin/activate

outfile=$(mktemp)
python src/main.py "$outfile" "$1" 1>&2
cat "$outfile"

deactivate
