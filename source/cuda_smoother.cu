/*************************************************************************
 * Copyright (c) 2017-2018 by the mfmg authors                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#include <mfmg/cuda_smoother.cuh>

#include <mfmg/cuda_matrix_operator.cuh>
#include <mfmg/dealii_operator_device_helpers.cuh>

namespace mfmg
{
namespace
{
template <typename VectorType>
struct SmootherOperator
{
  static void
  apply(SparseMatrixDevice<typename VectorType::value_type> const &matrix,
        SparseMatrixDevice<typename VectorType::value_type> const &smoother,
        VectorType const &b, VectorType &x);
};

template <typename VectorType>
void SmootherOperator<VectorType>::apply(
    SparseMatrixDevice<typename VectorType::value_type> const &matrix,
    SparseMatrixDevice<typename VectorType::value_type> const &smoother,
    VectorType const &b, VectorType &x)
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <>
void SmootherOperator<VectorDevice<double>>::apply(
    SparseMatrixDevice<double> const &matrix,
    SparseMatrixDevice<double> const &smoother, VectorDevice<double> const &b,
    VectorDevice<double> &x)
{
  // r = -(b - Ax)
  VectorDevice<double> r(b);
  matrix.vmult(r, x);
  r.add(-1., b);

  // x = x + B^{-1} (-r)
  VectorDevice<double> tmp(x);
  smoother.vmult(tmp, r);
  x.add(-1., tmp);
}

template <>
void SmootherOperator<dealii::LinearAlgebra::distributed::Vector<double>>::
    apply(SparseMatrixDevice<double> const &matrix,
          SparseMatrixDevice<double> const &smoother,
          dealii::LinearAlgebra::distributed::Vector<double> const &b,
          dealii::LinearAlgebra::distributed::Vector<double> &x)
{
  // Copy to the device
  VectorDevice<double> x_dev(x);
  VectorDevice<double> b_dev(b);

  SmootherOperator<VectorDevice<double>>::apply(matrix, smoother, b_dev, x_dev);

  // Move the data to the host
  std::vector<double> x_host(x.local_size());
  cuda_mem_copy_to_host(x_dev.val_dev, x_host);
  std::copy(x_host.begin(), x_host.end(), x.begin());
}

template <typename ScalarType>
__global__ void extract_inv_diag(ScalarType const *const matrix_value,
                                 int const *const matrix_column_index,
                                 int const *const matrix_row_index,
                                 int const size, ScalarType *value)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
    if (matrix_column_index[idx] == matrix_row_index[idx])
      value[matrix_column_index[idx]] = 1. / matrix_value[idx];
}
} // namespace

template <typename VectorType>
CudaSmoother<VectorType>::CudaSmoother(
    std::shared_ptr<Operator<VectorType> const> op,
    std::shared_ptr<boost::property_tree::ptree const> params)
    : Smoother<VectorType>(op, params)
{
  std::string prec_type = this->_params->get("smoother.type", "Jacobi");
  // Transform to lower case
  std::transform(prec_type.begin(), prec_type.end(), prec_type.begin(),
                 tolower);

  ASSERT_THROW(prec_type == "jacobi", "Only Jacobi smoother is implemented.");

  // Downcast the operator
  auto cuda_operator =
      std::dynamic_pointer_cast<CudaMatrixOperator<VectorType> const>(
          this->_operator);
  auto sparse_matrix = cuda_operator->get_matrix();

  ASSERT(sparse_matrix->m() == sparse_matrix->n(),
         "The matrix is not square. The matrix is a " +
             std::to_string(sparse_matrix->m()) + " by " +
             std::to_string(sparse_matrix->n()) + " .");

  // Extract diagonal elements
  unsigned int const size = sparse_matrix->n_local_rows();
  value_type *val_dev = nullptr;
  cuda_malloc(val_dev, size);
  int *column_index_dev = nullptr;
  cuda_malloc(column_index_dev, size);
  int *row_ptr_dev = nullptr;
  cuda_malloc(row_ptr_dev, size + 1);
  unsigned int const local_nnz = sparse_matrix->local_nnz();
  int *row_index_coo_dev = nullptr;
  cuda_malloc(row_index_coo_dev, local_nnz);

  // Change to COO format. The only thing that needs to be change to go from CSR
  // to COO is to change row_ptr_dev with row_index_coo_dev.
  cusparseStatus_t cusparse_error_code = cusparseXcsr2coo(
      sparse_matrix->cusparse_handle, sparse_matrix->row_ptr_dev, local_nnz,
      size, row_index_coo_dev, CUSPARSE_INDEX_BASE_ZERO);
  ASSERT_CUSPARSE(cusparse_error_code);

  int n_blocks = 1 + (local_nnz - 1) / block_size;
  extract_inv_diag<<<n_blocks, block_size>>>(
      sparse_matrix->val_dev, sparse_matrix->column_index_dev,
      row_index_coo_dev, local_nnz, val_dev);

  iota<<<n_blocks, block_size>>>(size, column_index_dev);

  n_blocks = 1 + size / block_size;
  iota<<<n_blocks, block_size>>>(size + 1, row_ptr_dev);

  _smoother.reinit(sparse_matrix->get_mpi_communicator(), val_dev,
                   column_index_dev, row_ptr_dev, size,
                   sparse_matrix->locally_owned_range_indices(),
                   sparse_matrix->locally_owned_range_indices(),
                   sparse_matrix->cusparse_handle);

  cuda_free(row_index_coo_dev);
}

template <typename VectorType>
void CudaSmoother<VectorType>::apply(VectorType const &b, VectorType &x) const
{
  auto cuda_operator =
      std::dynamic_pointer_cast<CudaMatrixOperator<VectorType> const>(
          this->_operator);
  auto matrix = cuda_operator->get_matrix();
  SmootherOperator<VectorType>::apply(*matrix, _smoother, b, x);
}
} // namespace mfmg

template class mfmg::CudaSmoother<mfmg::VectorDevice<double>>;
template class mfmg::CudaSmoother<
    dealii::LinearAlgebra::distributed::Vector<double>>;
