/*************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#define BOOST_TEST_MODULE amgx_direct_solver

#include "main.cc"

#if MFMG_WITH_AMGX
#include <mfmg/cuda/cuda_matrix_operator.cuh>
#include <mfmg/cuda/cuda_solver.cuh>
#include <mfmg/cuda/sparse_matrix_device.cuh>
#include <mfmg/cuda/utils.cuh>

#include <boost/property_tree/ptree.hpp>

#include <random>

#include <cusolverDn.h>
#include <cusolverSp.h>

BOOST_AUTO_TEST_CASE(amgx_1_proc)
{
  int comm_size = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  if (comm_size == 1)
  {
    mfmg::CudaHandle cuda_handle;

    // Create the matrix on the host.
    dealii::SparsityPattern sparsity_pattern;
    dealii::SparseMatrix<double> matrix;
    unsigned int const size = 3000;
    std::vector<std::vector<unsigned int>> column_indices(size);
    for (unsigned int i = 0; i < size; ++i)
    {
      unsigned int j_max = std::min(size, i + 2);
      unsigned int j_min = (i == 0) ? 0 : i - 1;
      for (unsigned int j = j_min; j < j_max; ++j)
        column_indices[i].emplace_back(j);
    }
    sparsity_pattern.copy_from(size, size, column_indices.begin(),
                               column_indices.end());
    matrix.reinit(sparsity_pattern);
    for (unsigned int i = 0; i < size; ++i)
    {
      unsigned int j_max = std::min(size - 1, i + 1);
      unsigned int j_min = (i == 0) ? 0 : i - 1;
      matrix.set(i, j_min, -1.);
      matrix.set(i, j_max, -1.);
      matrix.set(i, i, 4.);
    }

    // Generate a random solution and then compute the rhs
    dealii::Vector<double> sol_ref(size);
    std::default_random_engine generator;
    std::normal_distribution<> distribution(10., 2.);
    for (auto &val : sol_ref)
      val = distribution(generator);

    dealii::Vector<double> rhs(size);
    matrix.vmult(rhs, sol_ref);

    // Move the matrix and the rhs to the device
    auto matrix_dev = std::make_shared<mfmg::SparseMatrixDevice<double>>(
        mfmg::convert_matrix(matrix));
    matrix_dev->cusparse_handle = cuda_handle.cusparse_handle;
    cusparseStatus_t cusparse_error_code =
        cusparseCreateMatDescr(&matrix_dev->descr);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatType(matrix_dev->descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatIndexBase(matrix_dev->descr, CUSPARSE_INDEX_BASE_ZERO);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    auto partitioner =
        std::make_shared<dealii::Utilities::MPI::Partitioner>(size);
    dealii::LinearAlgebra::distributed::Vector<double,
                                               dealii::MemorySpace::CUDA>
        rhs_dev(partitioner);
    dealii::LinearAlgebra::distributed::Vector<double,
                                               dealii::MemorySpace::CUDA>
        solution_dev(partitioner);
    std::vector<double> rhs_host(size);
    std::copy(rhs.begin(), rhs.end(), rhs_host.begin());
    mfmg::cuda_mem_copy_to_dev(rhs_host, rhs_dev.get_values());
    auto params = std::make_shared<boost::property_tree::ptree>();

    params->put("solver.type", "amgx");
    params->put("solver.config_file", "amgx_config_fgmres.json");
    std::shared_ptr<mfmg::Operator<dealii::LinearAlgebra::distributed::Vector<
        double, dealii::MemorySpace::CUDA>>>
        op_dev = std::make_shared<
            mfmg::CudaMatrixOperator<dealii::LinearAlgebra::distributed::Vector<
                double, dealii::MemorySpace::CUDA>>>(matrix_dev);
    mfmg::CudaSolver<dealii::LinearAlgebra::distributed::Vector<
        double, dealii::MemorySpace::CUDA>>
        direct_solver_dev(cuda_handle, op_dev, params);
    direct_solver_dev.apply(rhs_dev, solution_dev);

    // Move the result back to the host
    int const n_local_rows = matrix_dev->n_local_rows();
    std::vector<double> solution_host(n_local_rows);
    mfmg::cuda_mem_copy_to_host(solution_dev.get_values(), solution_host);

    // Check the result
    for (unsigned int i = 0; i < n_local_rows; ++i)
      BOOST_CHECK_CLOSE(solution_host[i], sol_ref[i], 1e-7);
  }
}

BOOST_AUTO_TEST_CASE(amgx_2_procs)
{
  int n_devices = 0;
  cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
  mfmg::ASSERT_CUDA(cuda_error_code);
  int comm_size = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  if ((n_devices == 2) && (comm_size == 2))
  {
    int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (rank < 2)
    {
      cuda_error_code = cudaSetDevice(rank);

      mfmg::CudaHandle cuda_handle;

      // Create the matrix on the host.
      unsigned int const n_local_rows = 10000;
      unsigned int const row_offset = rank * n_local_rows;
      unsigned int const size = comm_size * n_local_rows;
      dealii::IndexSet parallel_partitioning(size);
      for (unsigned int i = 0; i < n_local_rows; ++i)
        parallel_partitioning.add_index(row_offset + i);
      parallel_partitioning.compress();
      dealii::TrilinosWrappers::SparseMatrix sparse_matrix(
          parallel_partitioning);

      for (unsigned int i = 0; i < n_local_rows; ++i)
      {
        unsigned int const row = row_offset + i;
        unsigned int j_max = std::min(size - 1, row + 1);
        unsigned int j_min = (row == 0) ? 0 : row - 1;
        sparse_matrix.set(row, j_min, -1.);
        sparse_matrix.set(row, j_max, -1.);
        sparse_matrix.set(row, row, 4.);
      }

      sparse_matrix.compress(dealii::VectorOperation::insert);

      // Generate a random solution and then compute the rhs
      auto range_indexset = sparse_matrix.locally_owned_range_indices();
      dealii::LinearAlgebra::distributed::Vector<double> sol_ref(
          range_indexset, MPI_COMM_WORLD);
      for (unsigned int i = 0; i < n_local_rows; ++i)
        sol_ref.local_element(i) = row_offset + i + 1;

      dealii::LinearAlgebra::distributed::Vector<double> rhs(range_indexset,
                                                             MPI_COMM_WORLD);
      sparse_matrix.vmult(rhs, sol_ref);

      // Move the matrix and the rhs to the device
      auto matrix_dev = std::make_shared<mfmg::SparseMatrixDevice<double>>(
          mfmg::convert_matrix(sparse_matrix));
      matrix_dev->cusparse_handle = cuda_handle.cusparse_handle;
      cusparseStatus_t cusparse_error_code =
          cusparseCreateMatDescr(&matrix_dev->descr);
      mfmg::ASSERT_CUSPARSE(cusparse_error_code);
      cusparse_error_code =
          cusparseSetMatType(matrix_dev->descr, CUSPARSE_MATRIX_TYPE_GENERAL);
      mfmg::ASSERT_CUSPARSE(cusparse_error_code);
      cusparse_error_code =
          cusparseSetMatIndexBase(matrix_dev->descr, CUSPARSE_INDEX_BASE_ZERO);
      mfmg::ASSERT_CUSPARSE(cusparse_error_code);
      auto rhs_dev = mfmg::copy_from_host(rhs);
      auto solution_dev = mfmg::copy_from_host(sol_ref);
      auto params = std::make_shared<boost::property_tree::ptree>();

      params->put("solver.type", "amgx");
      params->put("solver.config_file", "amgx_config_fgmres.json");

      std::shared_ptr<mfmg::Operator<dealii::LinearAlgebra::distributed::Vector<
          double, dealii::MemorySpace::CUDA>>>
      cuda_op(new mfmg::CudaMatrixOperator<
              dealii::LinearAlgebra::distributed::Vector<
                  double, dealii::MemorySpace::CUDA>>(matrix_dev));
      mfmg::CudaSolver<dealii::LinearAlgebra::distributed::Vector<
          double, dealii::MemorySpace::CUDA>>
          direct_solver_dev(cuda_handle, cuda_op, params);

      // Move the result back to the host
      std::vector<double> solution_host(n_local_rows);
      mfmg::cuda_mem_copy_to_host(solution_dev.get_values(), solution_host);

      for (unsigned int i = 0; i < n_local_rows; ++i)
        BOOST_CHECK_CLOSE(solution_host[i], sol_ref.local_element(i), 1e-7);
    }
  }
}
#endif
