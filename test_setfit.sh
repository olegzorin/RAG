#!/bin/zsh

pushd /Users/oleg/Projects/PPC-RAG/ || exit

source .venv/bin/activate

infile=$(mktemp)
outfile=$(mktemp)

trap 'rm -rf $infile $outfile' EXIT

cat - > "$infile"

python src/run_setfit.py "$infile" "$outfile"

deactivate

cat $outfile

popd