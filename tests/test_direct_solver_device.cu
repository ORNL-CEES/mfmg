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

#define BOOST_TEST_MODULE direct_solver_device

#include <mfmg/common/exceptions.hpp>
#include <mfmg/cuda/cuda_matrix_operator.cuh>
#include <mfmg/cuda/cuda_solver.cuh>
#include <mfmg/cuda/utils.cuh>

#include <random>

#include "main.cc"

BOOST_AUTO_TEST_CASE(direct_solver)
{
  mfmg::CudaHandle cuda_handle;

  // Create the matrix on the host.
  dealii::SparsityPattern sparsity_pattern;
  dealii::SparseMatrix<double> matrix;
  unsigned int const size = 30;
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

  // Move the matrix and the rhs to the host
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
  dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::CUDA>
      rhs_dev(partitioner);
  std::vector<double> rhs_host(size);
  std::copy(rhs.begin(), rhs.end(), rhs_host.begin());
  mfmg::cuda_mem_copy_to_dev(rhs_host, rhs_dev.get_values());
  auto params = std::make_shared<boost::property_tree::ptree>();

  for (auto solver : {"cholesky", "lu_dense", "lu_sparse_host"})
  {
    params->put("solver.type", solver);

    // Solve on the device
    std::shared_ptr<mfmg::Operator<dealii::LinearAlgebra::distributed::Vector<
        double, dealii::MemorySpace::CUDA>>>
    op_dev(
        new mfmg::CudaMatrixOperator<dealii::LinearAlgebra::distributed::Vector<
            double, dealii::MemorySpace::CUDA>>(matrix_dev));
    mfmg::CudaSolver<dealii::LinearAlgebra::distributed::Vector<
        double, dealii::MemorySpace::CUDA>>
        direct_solver_dev(cuda_handle, op_dev, params);
    dealii::LinearAlgebra::distributed::Vector<double,
                                               dealii::MemorySpace::CUDA>
        x_dev(partitioner);

    direct_solver_dev.apply(rhs_dev, x_dev);

    // Move the result back to the host
    std::vector<double> x_host(size);
    mfmg::cuda_mem_copy_to_host(x_dev.get_values(), x_host);

    // Check the result
    for (unsigned int i = 0; i < size; ++i)
      BOOST_CHECK_CLOSE(x_host[i], sol_ref[i], 1e-12);
  }
}
