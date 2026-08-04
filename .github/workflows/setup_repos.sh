# NOTE: the branches below differ from those the test suite runs against in
# test.yml (firedrake indiamai/fuse_mesh_cell, fiat indiamai/fuse). The docs
# are therefore built against a different dependency set from the one that is
# tested. Unify these once it is confirmed the docs build on the tested pair.

/usr/bin/git config --global --add safe.directory /opt/firedrake/
cd /opt/firedrake/
git fetch
git checkout indiamai/fuse
git pull
pip install pybind11 pyrsistent Cython
make

/usr/bin/git config --global --add safe.directory ~/
cd ~
git clone https://github.com/firedrakeproject/fiat.git
/usr/bin/git config --global --add safe.directory ~/fiat
cd fiat
git fetch
git checkout indiamai/fuse
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
