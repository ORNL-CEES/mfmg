Installation
============

This section provide guidelines for installing DataTransferKit and its TPLs.

Install third-party libraries
-----------------------------

The following third party libraries (TPLs) are used by mfmg:

+------------------------+-----------------------------------+
| Packages               | Version                           |
+========================+===================================+
| ARPACK                 | N/A                               |
+------------------------+-----------------------------------+
| Boost                  | 1.65.1                            |
+------------------------+-----------------------------------+
| BLAS/LAPACK            | N/A                               |
+------------------------+-----------------------------------+
| deal.II                | 8.5 (development for CUDA support)|
+------------------------+-----------------------------------+
| MPI                    | N/A                               |
+------------------------+-----------------------------------+
| Trilinos               | 12.X                              |
+------------------------+-----------------------------------+

The dependencies of mfmg may be built using `Spack
<https://github.com/llnl/spack>`_ package manager. You need to install the
following package:

.. code::

    $ spack install dealii

This will install all the dependencies of mfmg. If you want to use CUDA, you
will need to install the development version of deal.II

.. code::

    $ spack install dealii@develop


Building mfmg
-------------

Create a ``do-configure`` script such as:

.. code-block:: bash

    cmake \
        -D CMAKE_BUILD_TYPE=Release \
        -D MFMG_ENABLE_TESTS=ON \
        -D MFMG_ENABLE_CUDA=ON \
        -D CMAKE_CUDA_FLAGS="-arch=sm_35" \
        -D DEAL_II_DIR=${DEAL_II_DIR}
        ..

and run it from your build directory:

.. code::

    $ mkdir build && cd build
    $ ../do-configure
