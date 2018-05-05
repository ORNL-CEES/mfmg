Developer Tools
===============

Run MFMG development environment in a Docker container
------------------------------------------------------

To start a container from the MFMG pre-built Docker image that is used in the
automated build on Jenkins, run:

.. code:: bash

    [host]$ cd ci
    [host]$ echo COMPOSE_PROJECT_NAME=$USER > .env # [optional] specify a project name
    [host]$ docker-compose pull # pull the most up-to-date version of the MFMG base image
    [host]$ docker-compose up -d # start the container

This will mount the local MFMG source directory into the container.
We recommend you use a ``.env`` file to specify an alternate project name (the
default being the directory name, i.e. ``ci``).  This will let you run
multiple isolated environments on a single host.  Here the service name will be
prefixed by your username which will prevent interferences with other developers
on the same system.

Then to launch an interactive Bash session inside that container, do:

.. code:: bash

    [host]$ docker-compose exec mfmg_dev bash

Configure, build, and test as you would usually do:

.. code:: bash

    [container]$ ./ci/compile_and_run.sh .
    [container]$ cd build/
    [container]$ ctest -j<N>

Do not forget to cleanup after yourself:

.. code:: bash

    [container]$ exit
    [host]$ docker-compose down # stop and remove the container
