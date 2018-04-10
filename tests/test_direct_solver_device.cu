/*************************************************************************
 * Copyright (c) 2018 by the mfmg authors                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#define BOOST_TEST_MODULE direct_solver_device

#include "main.cc"

#include <mfmg/dealii_operator_device.cuh>
#include <mfmg/exceptions.hpp>
#include <mfmg/utils.cuh>

#include <random>

BOOST_AUTO_TEST_CASE(direct_solver)
{
  // Create the cusolver_dn_handle
  cusolverDnHandle_t cusolver_dn_handle = nullptr;
  cusolverStatus_t cusolver_error_code;
  cusolver_error_code = cusolverDnCreate(&cusolver_dn_handle);
  mfmg::ASSERT_CUSOLVER(cusolver_error_code);
  // Create the cusolver_sp_handle
  cusolverSpHandle_t cusolver_sp_handle = nullptr;
  cusolver_error_code = cusolverSpCreate(&cusolver_sp_handle);
  mfmg::ASSERT_CUSOLVER(cusolver_error_code);
  // Create the cusparse_handle
  cusparseHandle_t cusparse_handle = nullptr;
  cusparseStatus_t cusparse_error_code;
  cusparse_error_code = cusparseCreate(&cusparse_handle);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);

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
  mfmg::SparseMatrixDevice<double> matrix_dev(mfmg::convert_matrix(matrix));
  matrix_dev.cusparse_handle = cusparse_handle;
  cusparse_error_code = cusparseCreateMatDescr(&matrix_dev.descr);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code =
      cusparseSetMatType(matrix_dev.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code =
      cusparseSetMatIndexBase(matrix_dev.descr, CUSPARSE_INDEX_BASE_ZERO);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  auto partitioner =
      std::make_shared<dealii::Utilities::MPI::Partitioner>(size);
  mfmg::VectorDevice<double> rhs_dev(partitioner);
  std::vector<double> rhs_host(size);
  std::copy(rhs.begin(), rhs.end(), rhs_host.begin());
  mfmg::cuda_mem_copy_to_dev(rhs_host, rhs_dev.val_dev);

  for (auto solver : {"cholesky", "lu_dense", "lu_sparse_host"})
  {
    // Solve on the device
    mfmg::DirectDeviceOperator<mfmg::VectorDevice<double>> direct_solver_dev(
        cusolver_dn_handle, cusolver_sp_handle, matrix_dev, solver);
    BOOST_CHECK_EQUAL(direct_solver_dev.m(), matrix_dev.m());
    BOOST_CHECK_EQUAL(direct_solver_dev.n(), matrix_dev.n());
    mfmg::VectorDevice<double> x_dev(partitioner);

    direct_solver_dev.apply(rhs_dev, x_dev);

    // Move the result back to the host
    std::vector<double> x_host(size);
    mfmg::cuda_mem_copy_to_host(x_dev.val_dev, x_host);

    // Check the result
    for (unsigned int i = 0; i < size; ++i)
      BOOST_CHECK_CLOSE(x_host[i], sol_ref[i], 1e-12);
  }

  cusolver_error_code = cusolverDnDestroy(cusolver_dn_handle);
  mfmg::ASSERT_CUSOLVER(cusolver_error_code);
  cusolver_error_code = cusolverSpDestroy(cusolver_sp_handle);
  mfmg::ASSERT_CUSOLVER(cusolver_error_code);
  cusparse_error_code = cusparseDestroy(cusparse_handle);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
}
