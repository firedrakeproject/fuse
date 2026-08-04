:orphan: true

Installing FUSE
----------------

FUSE is currently under active development. The GitHub repository can be found `here <https://github.com/firedrakeproject/fuse>`_. If you would like to test it out, a preliminary release version can be installed with:

.. code-block:: bash

   pip install fuse-element

The current main branch version can be installed with

.. code-block:: bash

   pip install git+https://github.com/firedrakeproject/fuse.git

In order for this version of FUSE to function fully, it is necessary to use the correct branches for certain packages. These are the branches the test suite runs against, and are kept in step with ``.github/workflows/test.yml``:

.. code-block:: bash

   git+https://github.com/firedrakeproject/fiat.git@indiamai/fuse
   git+https://github.com/firedrakeproject/firedrake.git@indiamai/fuse_mesh_cell
