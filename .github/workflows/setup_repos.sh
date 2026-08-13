#!/bin/bash
# Branch names are defined once in .github/branches and shared with the test
# and smoke workflows so the docs build against the tested dependency set.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../branches"

/usr/bin/git config --global --add safe.directory /opt/firedrake/
cd /opt/firedrake/
git fetch
git checkout "$FIREDRAKE_BRANCH"
git pull
pip install pybind11 pyrsistent Cython
make

/usr/bin/git config --global --add safe.directory ~/
cd ~
git clone https://github.com/firedrakeproject/fiat.git
/usr/bin/git config --global --add safe.directory ~/fiat
cd fiat
git fetch
git checkout "$FIAT_BRANCH"
git status
python3 -m pip install --break-system-packages -e .

#/usr/bin/git config --global --add safe.directory ~
#cd ~
#git clone https://github.com/firedrakeproject/ufl.git
#/usr/bin/git config --global --add safe.directory ~/ufl
#cd ufl
#git fetch
#git checkout indiamai/integrate-fuse
#git status
#python3 -m pip install --break-system-packages -e .
