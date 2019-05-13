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

#define BOOST_TEST_MODULE smoother_device

#include <mfmg/common/exceptions.hpp>
#include <mfmg/cuda/cuda_matrix_operator.cuh>
#include <mfmg/cuda/cuda_smoother.cuh>
#include <mfmg/cuda/sparse_matrix_device.cuh>
#include <mfmg/cuda/utils.cuh>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <boost/property_tree/ptree.hpp>

#include "main.cc"

BOOST_AUTO_TEST_CASE(smoother)
{
  // Create the cusparse handle
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

  double constexpr scalar_value = 1.;
  std::vector<double> domain_host(size);
  for (auto &v : domain_host)
    v = scalar_value;
  std::vector<double> range_host(size, 0.);
  dealii::Vector<double> domain_vector(size);
  dealii::Vector<double> range_vector(size);
  for (auto &v : domain_vector)
    v = scalar_value;

  // Compute the reference solution
  dealii::PreconditionJacobi<dealii::SparseMatrix<double>> precondition;
  precondition.initialize(matrix);
  dealii::Vector<double> res(domain_vector);
  matrix.vmult(res, range_vector);
  res.add(-1., domain_vector);
  dealii::Vector<double> tmp(range_vector);
  precondition.vmult(tmp, res);
  range_vector.add(-1., tmp);

  // Move the matrix to the device
  auto matrix_dev = std::make_shared<mfmg::SparseMatrixDevice<double>>(
      mfmg::convert_matrix(matrix));
  matrix_dev->cusparse_handle = cusparse_handle;
  cusparse_error_code = cusparseCreateMatDescr(&matrix_dev->descr);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code =
      cusparseSetMatType(matrix_dev->descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code =
      cusparseSetMatIndexBase(matrix_dev->descr, CUSPARSE_INDEX_BASE_ZERO);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);

  // Build the smoother operator
  std::shared_ptr<mfmg::Operator<dealii::LinearAlgebra::distributed::Vector<
      double, dealii::MemorySpace::CUDA>>>
  cuda_op(
      new mfmg::CudaMatrixOperator<dealii::LinearAlgebra::distributed::Vector<
          double, dealii::MemorySpace::CUDA>>(matrix_dev));
  auto param = std::make_shared<boost::property_tree::ptree>();
  mfmg::CudaSmoother<dealii::LinearAlgebra::distributed::Vector<
      double, dealii::MemorySpace::CUDA>>
      smoother_operator(cuda_op, param);

  // Apply the smoother
  auto domain_dev = cuda_op->build_domain_vector();
  auto range_dev = cuda_op->build_range_vector();
  mfmg::cuda_mem_copy_to_dev(domain_host, domain_dev->get_values());
  mfmg::cuda_mem_copy_to_dev(range_host, range_dev->get_values());
  smoother_operator.apply(*domain_dev, *range_dev);

  // Compare the solution
  mfmg::cuda_mem_copy_to_host(range_dev->get_values(), range_host);
  for (unsigned int i = 0; i < size; ++i)
    BOOST_CHECK_CLOSE(range_host[i], range_vector[i], 1e-12);

  // Destroy the cusparse handle
  cusparse_error_code = cusparseDestroy(cusparse_handle);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
}
