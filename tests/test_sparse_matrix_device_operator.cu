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

#define BOOST_TEST_MODULE sparse_matrix_device_operator

#include <mfmg/cuda/cuda_matrix_operator.cuh>
#include <mfmg/cuda/sparse_matrix_device.cuh>
#include <mfmg/cuda/utils.cuh>

#include <set>
#include <utility>

#include "main.cc"

BOOST_AUTO_TEST_CASE(matrix_operator)
{
  // Create the cusparse_handle
  cusparseHandle_t cusparse_handle = nullptr;
  cusparseStatus_t cusparse_error_code;
  cusparse_error_code = cusparseCreate(&cusparse_handle);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);

  unsigned int const n_rows = 30;
  unsigned int const nnz_per_row = 10;
  unsigned int const n_cols = n_rows + nnz_per_row - 1;
  dealii::SparsityPattern sparsity_pattern;
  std::vector<std::vector<unsigned int>> column_indices(
      n_rows, std::vector<unsigned int>(nnz_per_row, 0));
  std::set<std::pair<unsigned int, unsigned int>> sparsity_pattern_indices;
  for (unsigned int i = 0; i < n_rows; ++i)
    for (unsigned int j = 0; j < nnz_per_row; ++j)
    {
      column_indices[i][j] = i + j;
      sparsity_pattern_indices.insert(std::make_pair(i, i + j));
    }
  sparsity_pattern.copy_from(n_rows, n_cols, column_indices.begin(),
                             column_indices.end());
  dealii::SparseMatrix<double> sparse_matrix(sparsity_pattern);
  for (unsigned int i = 0; i < n_rows; ++i)
    for (unsigned int j = 0; j < n_cols; ++j)
      if (sparsity_pattern_indices.count(std::make_pair(i, j)) > 0)
        sparse_matrix.set(i, j, static_cast<double>(i + j));

  auto sparse_matrix_dev = std::make_shared<mfmg::SparseMatrixDevice<double>>(
      mfmg::convert_matrix(sparse_matrix));
  sparse_matrix_dev->cusparse_handle = cusparse_handle;
  cusparse_error_code = cusparseCreateMatDescr(&sparse_matrix_dev->descr);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code = cusparseSetMatType(sparse_matrix_dev->descr,
                                           CUSPARSE_MATRIX_TYPE_GENERAL);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code = cusparseSetMatIndexBase(sparse_matrix_dev->descr,
                                                CUSPARSE_INDEX_BASE_ZERO);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  mfmg::CudaMatrixOperator<dealii::LinearAlgebra::distributed::Vector<
      double, dealii::MemorySpace::CUDA>>
      matrix_operator(sparse_matrix_dev);

  // Check build_domain_vector
  auto domain_dev = matrix_operator.build_domain_vector();
  BOOST_CHECK_EQUAL(domain_dev->get_partitioner()->size(), n_cols);

  // Check build_range_vector
  auto range_dev = matrix_operator.build_range_vector();
  BOOST_CHECK_EQUAL(range_dev->get_partitioner()->size(), n_rows);

  // Check apply
  std::vector<double> domain_vector(n_cols, 1.);
  mfmg::cuda_mem_copy_to_dev(domain_vector, domain_dev->get_values());
  matrix_operator.apply(*domain_dev, *range_dev);
  dealii::Vector<double> domain_dealii(domain_vector.begin(),
                                       domain_vector.end());
  dealii::Vector<double> range_dealii(n_rows);
  sparse_matrix.vmult(range_dealii, domain_dealii);
  std::vector<double> range_vector(n_rows);
  mfmg::cuda_mem_copy_to_host(range_dev->get_values(), range_vector);
  for (unsigned int i = 0; i < n_rows; ++i)
    BOOST_CHECK_EQUAL(range_vector[i], range_dealii[i]);

  // Check transpose
  dealii::SparsityPattern transpose_sparsity_pattern;
  std::vector<std::vector<unsigned int>> transpose_column_indices(n_cols);
  for (unsigned int i = 0; i < n_cols; ++i)
    for (unsigned int j = 0; j < n_rows; ++j)
      if (sparsity_pattern_indices.count(std::make_pair(j, i)) > 0)
        transpose_column_indices[i].emplace_back(j);
  transpose_sparsity_pattern.copy_from(n_cols, n_rows,
                                       transpose_column_indices.begin(),
                                       transpose_column_indices.end());
  dealii::SparseMatrix<double> transpose_sparse_matrix(
      transpose_sparsity_pattern);
  for (unsigned int i = 0; i < n_cols; ++i)
    for (unsigned int j = 0; j < n_rows; ++j)
      if (sparsity_pattern_indices.count(std::make_pair(j, i)) > 0)
        transpose_sparse_matrix.set(i, j, static_cast<double>(i + j));
  auto transpose_matrix_operator = matrix_operator.transpose();
  // We don't have access to the underlying matrix directly so we apply the new
  // operator and check the results
  auto transpose_domain_dev = transpose_matrix_operator->build_domain_vector();
  auto transpose_range_dev = transpose_matrix_operator->build_range_vector();
  std::vector<double> transpose_domain_vector(n_rows, 1.);
  mfmg::cuda_mem_copy_to_dev(transpose_domain_vector,
                             transpose_domain_dev->get_values());
  transpose_matrix_operator->apply(*transpose_domain_dev, *transpose_range_dev);
  dealii::Vector<double> transpose_domain_dealii(
      transpose_domain_vector.begin(), transpose_domain_vector.end());
  dealii::Vector<double> transpose_range_dealii(n_cols);
  transpose_sparse_matrix.vmult(transpose_range_dealii,
                                transpose_domain_dealii);
  std::vector<double> transpose_range_vector(n_cols);
  mfmg::cuda_mem_copy_to_host(transpose_range_dev->get_values(),
                              transpose_range_vector);
  for (unsigned int i = 0; i < n_rows; ++i)
    BOOST_CHECK_EQUAL(transpose_range_vector[i], transpose_range_dealii[i]);

  // Check multiply
  sparse_matrix.vmult(range_dealii, transpose_range_dealii);
  auto multiplied_matrix_operator =
      matrix_operator.multiply(transpose_matrix_operator);
  multiplied_matrix_operator->apply(*transpose_domain_dev, *range_dev);

  mfmg::cuda_mem_copy_to_host(range_dev->get_values(), range_vector);

  for (unsigned int i = 0; i < n_rows; ++i)
    BOOST_CHECK_EQUAL(range_vector[i], range_dealii[i]);

  cusparse_error_code = cusparseDestroy(cusparse_handle);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
}
