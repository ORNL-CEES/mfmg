FROM nvidia/cuda:9.0-devel

ARG N_PROCS=16

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
      gcc \
      gfortran \
      build-essential \
      wget \
      curl \
      environment-modules \
      bison \
      python \
      git \
      lcov \
      zlib1g-dev \
      python2.7-dev \
      && \
      echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main" > /etc/apt/sources.list.d/llvm.list && \
      wget -q -O - http://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
      apt-get update && apt-get install -y \
      clang-format-6.0 \
      && \
      apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PREFIX=/scratch
ENV ARCHIVE_DIR=${PREFIX}/archive
ENV SOURCE_DIR=${PREFIX}/source
ENV BUILD_DIR=${PREFIX}/build
ENV INSTALL_DIR=/opt

RUN mkdir -p ${PREFIX} && \
    cd ${PREFIX} && \
    mkdir archive && \
    mkdir source && \
    mkdir build

RUN export CMAKE_VERSION=3.14.3 && \
    export CMAKE_VERSION_SHORT=3.14 && \
    export CMAKE_URL=https://cmake.org/files/v${CMAKE_VERSION_SHORT}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    export CMAKE_ARCHIVE=${ARCHIVE_DIR}/cmake.tar.gz && \
    export CMAKE_BUILD_DIR=${BUILD_DIR}/cmake && \
    wget --quiet ${CMAKE_URL} --output-document=${CMAKE_ARCHIVE}  && \
    mkdir -p ${CMAKE_BUILD_DIR} && \
    tar xf ${CMAKE_ARCHIVE} -C ${CMAKE_BUILD_DIR} --strip-components=1 && \
    mv ${CMAKE_BUILD_DIR} ${INSTALL_DIR} && \
    rm -rf ${CMAKE_ARCHIVE} && \
    rm -rf ${CMAKE_BUILD_DIR}
ENV PATH=${INSTALL_DIR}/cmake/bin:$PATH

# Install OpenMPI
RUN export OPENMPI_VERSION=4.0.1 && \
    export OPENMPI_VERSION_SHORT=4.0 && \
    export OPENMPI_SHA1=35bf7c9162b08ecdc4876af573786cd290015631 && \
    export OPENMPI_URL=https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSION_SHORT}/downloads/openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    export OPENMPI_ARCHIVE=${ARCHIVE_DIR}/openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    export OPENMPI_SOURCE_DIR=${SOURCE_DIR}/openmpi && \
    export OPENMPI_BUILD_DIR=${BUILD_DIR}/openmpi && \
    export OPENMPI_INSTALL_DIR=${INSTALL_DIR}/openmpi && \
    wget --quiet ${OPENMPI_URL} --output-document=${OPENMPI_ARCHIVE} && \
    echo "${OPENMPI_SHA1} ${OPENMPI_ARCHIVE}" | sha1sum -c && \
    mkdir -p ${OPENMPI_SOURCE_DIR} && \
    tar -xf ${OPENMPI_ARCHIVE} -C ${OPENMPI_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${OPENMPI_BUILD_DIR} && \
    cd ${OPENMPI_BUILD_DIR} && \
    ${OPENMPI_SOURCE_DIR}/configure --with-cuda --prefix=${OPENMPI_INSTALL_DIR} && \
    make -j${N_PROCS} install && \
    rm -rf ${OPENMPI_ARCHIVE} && \
    rm -rf ${OPENMPI_BUILD_DIR} && \
    rm -rf ${OPENMPI_SOURCE_DIR}
# Put OPENMPI_DIR at the end of the path so that /usr/local/bin/mpiexec will
# overwrite it
ENV PATH=$PATH:${INSTALL_DIR}/openmpi/bin

# Install AMGX
RUN export AMGX_HASH=6cb23fed26602e4873d5c1deb694a2c8480feac3 && \
    export AMGX_URL=https://github.com/nvidia/amgx/archive/${AMGX_HASH}.tar.gz && \
    export AMGX_ARCHIVE=${ARCHIVE_DIR}/amgx.tar.gz && \
    export AMGX_SOURCE_DIR=${SOURCE_DIR}/amgx && \
    export AMGX_BUILD_DIR=${BUILD_DIR}/amgx && \
    export AMGX_INSTALL_DIR=${INSTALL_DIR}/amgx && \
    wget --quiet ${AMGX_URL} --output-document=${AMGX_ARCHIVE} && \
    mkdir -p ${AMGX_SOURCE_DIR} && \
    tar -xf ${AMGX_ARCHIVE} -C ${AMGX_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${AMGX_BUILD_DIR} && cd ${AMGX_BUILD_DIR} && \
    cmake -DCMAKE_INSTALL_PREFIX=${AMGX_INSTALL_DIR} \
        ${AMGX_SOURCE_DIR} && \
    make -j${N_PROCS} install && \
    rm -rf ${AMGX_ARCHIVE} && \
    rm -rf ${AMGX_BUILD_DIR} && \
    rm -rf ${AMGX_SOURCE_DIR}
ENV AMGX_DIR=${INSTALL_DIR}/amgx

# Install Boost
RUN export BOOST_VERSION=1.67.0 && \
    export BOOST_VERSION_U=1_67_0 && \
    export BOOST_URL=https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_U}.tar.bz2 && \
    export BOOST_SHA256=2684c972994ee57fc5632e03bf044746f6eb45d4920c343937a465fd67a5adba && \
    export BOOST_ARCHIVE=${ARCHIVE_DIR}/boost_${BOOST_VERSION_U}.tar.bz2 && \
    export BOOST_SOURCE_DIR=${SOURCE_DIR}/boost && \
    export BOOST_BUILD_DIR=${BUILD_DIR}/boost && \
    export BOOST_INSTALL_DIR=${INSTALL_DIR}/boost && \
    wget --quiet ${BOOST_URL} --output-document=${BOOST_ARCHIVE} && \
    echo "${BOOST_SHA256} ${BOOST_ARCHIVE}" | sha256sum -c && \
    mkdir -p ${BOOST_SOURCE_DIR} && \
    tar -xf ${BOOST_ARCHIVE} -C ${BOOST_SOURCE_DIR} --strip-components=1 && \
    cd ${BOOST_SOURCE_DIR} && \
    ./bootstrap.sh \
        --prefix=${BOOST_INSTALL_DIR} \
        && \
    echo "using mpi ;" >> project-config.jam && \
    ./b2 -j${N_PROCS}\
        --build-dir=${BOOST_BUILD_DIR} \
        hardcode-dll-paths=true dll-path=${BOOST_INSTALL_DIR}/lib \
        link=shared \
        variant=release \
        install \
        && \
    rm -rf ${BOOST_ARCHIVE} && \
    rm -rf ${BOOST_BUILD_DIR} && \
    rm -rf ${BOOST_SOURCE_DIR}
ENV BOOST_ROOT=${INSTALL_DIR}/boost

# Install OpenBLAS (BLAS, Lapack, and Lapacke)
RUN export OPENBLAS_VERSION=0.3.6 && \
    export OPENBLAS_URL=https://github.com/xianyi/OpenBLAS/archive/v${OPENBLAS_VERSION}.tar.gz && \
    export OPENBLAS_SHA256=e64c8fe083832ffbc1459ab6c72f71d53afd3b36e8497c922a15a06b72e9002f && \
    export OPENBLAS_ARCHIVE=${ARCHIVE}/openblas.tar.gz && \
    export OPENBLAS_SOURCE_DIR=${SOURCE_DIR}/OpenBLAS-${OPENBLAS_VERSION} && \
    export OPENBLAS_BUILD_DIR=${BUILD_DIR}/openblas && \
    export OPENBLAS_INSTALL_DIR=${INSTALL_DIR}/openblas && \
    wget --quiet ${OPENBLAS_URL} --output-document=${OPENBLAS_ARCHIVE} && \
    echo "${OPENBLAS_SHA256} ${OPENBLAS_ARCHIVE}" | sha256sum -c && \
    mkdir -p ${OPENBLAS_SOURCE_DIR} && \
    tar -xf ${OPENBLAS_ARCHIVE} -C ${OPENBLAS_SOURCE_DIR} --strip-components=1 && \
    cd ${OPENBLAS_SOURCE_DIR} && \
    make -j${N_PROCS} DYNAMIC_ARCH=1 && \
    make install PREFIX=${OPENBLAS_INSTALL_DIR} && \
    rm -rf ${OPENBLAS_ARCHIVE} && \
    rm -rf ${OPENBLAS_BUILD_DIR} && \
    rm -rf ${OPENBLAS_SOURCE_DIR}
ENV OPENBLAS_DIR=${INSTALL_DIR}/openblas

# Install Trilinos 12.14.1
RUN export TRILINOS_HASH=316ed8236deebb613301e41bcf901aebc08cee74 && \
    export TRILINOS_URL=https://github.com/trilinos/Trilinos/archive/${TRILINOS_HASH}.tar.gz && \
    export TRILINOS_ARCHIVE=${ARCHIVE_DIR}/trilinos.tar.gz && \
    export TRILINOS_SOURCE_DIR=${SOURCE_DIR}/trilinos && \
    export TRILINOS_BUILD_DIR=${BUILD_DIR}/trilinos && \
    export TRILINOS_INSTALL_DIR=${INSTALL_DIR}/trilinos && \
    wget --quiet ${TRILINOS_URL} --output-document=${TRILINOS_ARCHIVE} && \
    mkdir -p ${TRILINOS_SOURCE_DIR} && \
    tar -xf ${TRILINOS_ARCHIVE} -C ${TRILINOS_SOURCE_DIR} --strip-components=1 && \
    mkdir ${TRILINOS_BUILD_DIR} && \
    cd ${TRILINOS_BUILD_DIR} && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=ON \
        -DTPL_ENABLE_MPI=ON \
        -DTPL_ENABLE_BLAS=ON \
        -DTPL_ENABLE_LAPACK=ON \
        -DTPL_ENABLE_Boost=ON \
        -DBoost_INCLUDE_DIRS=${BOOST_ROOT}/include \
        -DTPL_ENABLE_BoostLib=ON \
        -DBoostLib_INCLUDE_DIRS=${BOOST_ROOT}/include \
        -DBoostLib_LIBRARY_DIRS=${BOOST_ROOT}/lib \
        -DTrilinos_ENABLE_ALL_PACKAGES=OFF \
        -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
        -DTrilinos_ENABLE_TESTS=OFF \
        -DTrilinos_ENABLE_EXAMPLES=OFF \
        -DTrilinos_ENABLE_Anasazi=ON \
        -DAnasazi_ENABLE_TEUCHOS_TIME_MONITOR=OFF \
        -DTrilinos_ENABLE_Amesos=ON \
        -DTrilinos_ENABLE_AztecOO=ON \
        -DTrilinos_ENABLE_Epetra=ON \
        -DTrilinos_ENABLE_EpetraExt=ON \
        -DTrilinos_ENABLE_Ifpack=ON \
        -DTrilinos_ENABLE_ML=ON \
        -DTrilinos_ENABLE_MueLu=ON \
        -DTrilinos_ENABLE_Zoltan=ON \
        -DTrilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
        -DBLAS_LIBRARY_NAMES="openblas" \
        -DBLAS_LIBRARY_DIRS=${OPENBLAS_DIR}/lib \
        -DLAPACK_LIBRARY_NAMES="openblas" \
        -DLAPACK_LIBRARY_DIRS=${OPENBLAS_DIR}/lib \
        -DCMAKE_INSTALL_PREFIX=${TRILINOS_INSTALL_DIR} \
        ${TRILINOS_SOURCE_DIR} && \
    make -j${N_PROCS} install && \
    rm -rf ${TRILINOS_ARCHIVE} && \
    rm -rf ${TRILINOS_BUILD_DIR} && \
    rm -rf ${TRILINOS_SOURCE_DIR}
ENV TRILINOS_DIR=${INSTALL_DIR}/trilinos

# Install p4est
RUN export P4EST_VERSION=2.2 && \
    export P4EST_URL=http://p4est.github.io/release/p4est-${P4EST_VERSION}.tar.gz && \
    export P4EST_ARCHIVE=${ARCHIVE_DIR}/p4est-${P4EST_VERSION}.tar.gz && \
    export P4EST_SOURCE_DIR=${SOURCE_DIR}/p4est && \
    export P4EST_BUILD_DIR=${BUILD_DIR}/p4est && \
    export P4EST_INSTALL_DIR=${INSTALL_DIR}/p4est && \
    wget --quiet ${P4EST_URL} --output-document=${P4EST_ARCHIVE} && \
    mkdir -p ${P4EST_SOURCE_DIR} && \
    cd ${P4EST_SOURCE_DIR} && \
    wget --quiet https://www.dealii.org/9.0.0/external-libs/p4est-setup.sh && \
    bash ./p4est-setup.sh ${P4EST_ARCHIVE} ${P4EST_INSTALL_DIR} && \
    rm -rf ${P4EST_ARCHIVE} && \
    rm -rf ${P4EST_BUILD_DIR} && \
    rm -rf ${P4EST_SOURCE_DIR}
ENV P4EST_DIR=${INSTALL_DIR}/p4est

# Install Arpack-NG
RUN export ARPACK_VERSION=3.7.0 && \
    export ARPACK_URL=https://github.com/opencollab/arpack-ng/archive/${ARPACK_VERSION}.tar.gz && \
    export ARPACK_ARCHIVE=${ARCHIVE_DIR}/arpack-${ARPACK_VERSION}.tar.gz && \
    export ARPACK_SOURCE_DIR=${SOURCE_DIR}/arpack && \
    export ARPACK_BUILD_DIR=${BUILD_DIR}/arpack && \
    export ARPACK_INSTALL_DIR=${INSTALL_DIR}/arpack && \
    wget --quiet ${ARPACK_URL} --output-document=${ARPACK_ARCHIVE} && \
    mkdir -p ${ARPACK_SOURCE_DIR} && \
    tar -xf ${ARPACK_ARCHIVE} -C ${ARPACK_SOURCE_DIR} --strip-components=1 && \
    mkdir ${ARPACK_BUILD_DIR} && \
    cd ${ARPACK_BUILD_DIR} && \
    cmake -DMPI=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX=${ARPACK_INSTALL_DIR} \
        -DBLAS_LIBRARIES=${OPENBLAS_DIR}/lib/libopenblas.so \
        -DLAPACK_DIR=${OPENBLAS_DIR} \
        ${ARPACK_SOURCE_DIR} && \
    make -j${N_PROCS} install && \
    rm -rf ${ARPACK_ARCHIVE} && \
    rm -rf ${ARPACK_BUILD_DIR} && \
    rm -rf ${ARPACK_SOURCE_DIR}
ENV ARPACK_DIR=${INSTALL_DIR}/arpack

# Install deal.II
RUN export DEAL_II_HASH=89057dff516849b67f737355ff697f6407acc579 && \
    export DEAL_II_URL=https://github.com/dealii/dealii/archive/${DEAL_II_HASH}.tar.gz && \
    export DEAL_II_ARCHIVE=${ARCHIVE_DIR}/dealii.tar.gz && \
    export DEAL_II_SOURCE_DIR=${SOURCE_DIR}/dealii && \
    export DEAL_II_BUILD_DIR=${BUILD_DIR}/dealii && \
    export DEAL_II_INSTALL_DIR=${INSTALL_DIR}/dealii && \
    wget --quiet ${DEAL_II_URL} --output-document=${DEAL_II_ARCHIVE} && \
    mkdir -p ${DEAL_II_SOURCE_DIR} && \
    tar -xf ${DEAL_II_ARCHIVE} -C ${DEAL_II_SOURCE_DIR} --strip-components=1 && \
    cd ${DEAL_II_SOURCE_DIR} && wget --quiet https://raw.githubusercontent.com/ORNL-CEES/mfmg/master/scripts/0001-Enable-ARPACK-in-regular-mode.patch && \
    git apply 0001-Enable-ARPACK-in-regular-mode.patch && \
    mkdir -p ${DEAL_II_BUILD_DIR} && cd ${DEAL_II_BUILD_DIR} && \
    cmake -DCMAKE_BUILD_TYPE=DebugRelease \
        -DDEAL_II_WITH_MPI=ON \
        -DDEAL_II_WITH_LAPACK=ON \
        -DLAPACK_DIR=${OPENBLAS_DIR} \
        -DDEAL_II_WITH_P4EST=ON \
        -DP4EST_DIR=${P4EST_DIR} \
        -DDEAL_II_WITH_TRILINOS=ON \
        -DTRILINOS_DIR=${TRILINOS_DIR} \
        -DDEAL_II_WITH_ARPACK=ON \
        -DARPACK_DIR=${ARPACK_DIR} \
        -DDEAL_II_WITH_CUDA=ON \
        -DDEAL_II_WITH_CXX14=ON \
        -DDEAL_II_WITH_CXX17=OFF \
        -DDEAL_II_CUDA_FLAGS="-arch=sm_35" \
        -DDEAL_II_COMPONENT_EXAMPLES=OFF \
        -DCMAKE_INSTALL_PREFIX=${DEAL_II_INSTALL_DIR} \
        ${DEAL_II_SOURCE_DIR} && \
    make -j${N_PROCS} install && \
    rm -rf ${DEAL_II_ARCHIVE} && \
    rm -rf ${DEAL_II_BUILD_DIR} && \
    rm -rf ${DEAL_II_SOURCE_DIR}
ENV DEAL_II_DIR=${INSTALL_DIR}/dealii

# deal.II does not export the include directory for BLAS/LAPACK, so add it to
# CPATH
ENV CPATH=${OPENBLAS_DIR}/include:${CPATH}

# Append the option flag --allow-run-as-root to mpiexec
RUN echo '#!/usr/bin/env bash' > /usr/local/bin/mpiexec && \
    echo '${INSTALL_DIR}/openmpi/bin/mpiexec --allow-run-as-root "$@"' >> /usr/local/bin/mpiexec && \
    chmod +x /usr/local/bin/mpiexec
