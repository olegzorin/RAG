#!/bin/zsh

export PPC_HOME=${PPC_HOME:-/opt/greenxserver}

cd /Users/oleg/Projects/PPC-RAG/

source .venv/bin/activate

outfile=$(mktemp)
python src/main.py "$outfile" "$1" 1>&2
cat "$outfile"

deactivate
