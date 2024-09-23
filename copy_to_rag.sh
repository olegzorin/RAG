#!/bin/zsh

pushd src || exit 1

/Users/oleg/Projects/PPC/util/scp-to.sh rag main.py
/Users/oleg/Projects/PPC/util/scp-to.sh rag core.py
/Users/oleg/Projects/PPC/util/scp-to.sh rag conf.py
/Users/oleg/Projects/PPC/util/scp-to.sh rag vector.py
/Users/oleg/Projects/PPC/util/scp-to.sh rag graph.py
/Users/oleg/Projects/PPC/util/scp-to.sh rag db.py
/Users/oleg/Projects/PPC/util/scp-to.sh rag reader.py

popd || exit