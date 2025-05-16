/usr/bin/git config --global --add safe.directory /opt/firedrake/
cd /opt/firedrake/
git fetch
git checkout indiamai/fuse-quads
git pull
pip install pybind11 pyrsistent
make

/usr/bin/git config --global --add safe.directory ~/
cd ~
git clone https://github.com/firedrakeproject/fiat.git
/usr/bin/git config --global --add safe.directory ~/fiat
cd fiat
git fetch
git checkout indiamai/integrate_fuse
git status
python3 -m pip install --break-system-packages -e .

/usr/bin/git config --global --add safe.directory ~
cd ~
git clone https://github.com/firedrakeproject/ufl.git
/usr/bin/git config --global --add safe.directory ~/ufl
cd ufl
git fetch
git checkout indiamai/integrate-fuse
git status
python3 -m pip install --break-system-packages -e .