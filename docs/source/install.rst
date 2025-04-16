:orphan: true

Installing FUSE
----------------

FUSE is currently under active development. The GitHub repository can be found `here <https://github.com/firedrakeproject/fuse>`_. If you would like to test it out, a preliminary release version can be installed with:

.. code-block:: bash

   pip install fuse-element

The current main branch version can be installed with

.. code-block:: bash

   pip install git+https://github.com/indiamai/fuse.git

In order for this version of FUSE to function fully, it is necessary to use the correct branches for certain packages:

.. code-block:: bash

   git+https://github.com/firedrakeproject/fiat.git@indiamai/integrate_fuse
   git+https://github.com/firedrakeproject/firedrake.git@indiamai/new_def_integration
