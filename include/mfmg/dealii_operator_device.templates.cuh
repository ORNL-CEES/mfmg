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

#ifndef MFMG_DEALII_OPERATOR_DEVICE_TEMPLATES_CUH
#define MFMG_DEALII_OPERATOR_DEVICE_TEMPLATES_CUH

#include <mfmg/dealii_operator_device.cuh>
#include <mfmg/dealii_operator_device_helpers.cuh>
#include <mfmg/utils.cuh>

#include <EpetraExt_Transpose_RowMatrix.h>

#include <algorithm>

#define BLOCK_SIZE 512

namespace mfmg
{
template <typename VectorType>
SparseMatrixDeviceOperator<VectorType>::SparseMatrixDeviceOperator(
    std::shared_ptr<SparseMatrixDevice<typename VectorType::value_type>> matrix)
    : _matrix(matrix)
{
  ASSERT(matrix != nullptr, "The matrix must exist");
}

template <typename VectorType>
void SparseMatrixDeviceOperator<VectorType>::apply(VectorType const &x,
                                                   VectorType &y) const
{
  _matrix->vmult(y, x);
}

template <typename VectorType>
std::shared_ptr<MatrixOperator<VectorType>>
SparseMatrixDeviceOperator<VectorType>::transpose() const
{
  // TODO: to do this on the GPU
  // Copy the data to the cpu and then, use the trilinos to compute the
  // transpose. This is not the most efficient way to do this but it is the
  // easiest.
  unsigned int const local_nnz = _matrix->local_nnz();
  unsigned int const n_local_rows = _matrix->n_local_rows();
  std::vector<value_type> values(local_nnz);
  std::vector<int> column_index(local_nnz);
  std::vector<int> row_ptr(n_local_rows + 1);

  // Copy the data to the host
  cuda_mem_copy_to_host(_matrix->val_dev, values);
  cuda_mem_copy_to_host(_matrix->column_index_dev, column_index);
  cuda_mem_copy_to_host(_matrix->row_ptr_dev, row_ptr);

  // Create the sparse matrix on the host
  dealii::IndexSet locally_owned_rows = _matrix->locally_owned_range_indices();
  dealii::TrilinosWrappers::SparseMatrix sparse_matrix(
      locally_owned_rows, _matrix->locally_owned_domain_indices(),
      _matrix->get_mpi_communicator());

  std::vector<unsigned int> rows;
  locally_owned_rows.fill_index_vector(rows);
  for (unsigned int i = 0; i < n_local_rows; ++i)
  {
    unsigned int const row = rows[i];
    // deal.II has functions that use pointers but the pointers need to be
    // unsigned int instead of int. To avoid any problem, we set the elements
    // one by one which makes sure that the cast is correct.
    for (unsigned int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
      sparse_matrix.set(row, column_index[j], values[j]);
  }
  sparse_matrix.compress(dealii::VectorOperation::insert);

  // Transpose the sparse matrix
  auto epetra_matrix = sparse_matrix.trilinos_matrix();

  EpetraExt::RowMatrix_Transpose transposer;
  auto transposed_epetra_matrix =
      dynamic_cast<Epetra_CrsMatrix &>(transposer(epetra_matrix));

  auto transposed_matrix =
      std::make_shared<matrix_type>(convert_matrix(transposed_epetra_matrix));
  transposed_matrix->cusparse_handle = _matrix->cusparse_handle;
  cusparseStatus_t cusparse_error_code;
  cusparse_error_code = cusparseCreateMatDescr(&transposed_matrix->descr);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code = cusparseSetMatType(transposed_matrix->descr,
                                           CUSPARSE_MATRIX_TYPE_GENERAL);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code = cusparseSetMatIndexBase(transposed_matrix->descr,
                                                CUSPARSE_INDEX_BASE_ZERO);
  ASSERT_CUSPARSE(cusparse_error_code);

  return std::make_shared<SparseMatrixDeviceOperator<VectorType>>(
      transposed_matrix);
}

template <typename VectorType>
std::shared_ptr<MatrixOperator<VectorType>>
SparseMatrixDeviceOperator<VectorType>::multiply(
    MatrixOperator<VectorType> const &operator_b) const
{
  // Downcast to SparseMatrixDeviceOperator
  auto downcast_operator_b =
      static_cast<SparseMatrixDeviceOperator<VectorType> const &>(operator_b);

  auto a = this->get_matrix();
  auto b = downcast_operator_b.get_matrix();

  // Initialize c
  auto c = std::make_shared<matrix_type>(*a);

  a->mmult(*c, *b);

  return std::make_shared<SparseMatrixDeviceOperator<VectorType>>(c);
}

template <typename VectorType>
std::shared_ptr<VectorType>
SparseMatrixDeviceOperator<VectorType>::build_domain_vector() const
{
  auto partitioner =
      std::make_shared<const dealii::Utilities::MPI::Partitioner>(
          _matrix->locally_owned_domain_indices(),
          _matrix->get_mpi_communicator());

  return std::make_shared<vector_type>(partitioner);
}

template <typename VectorType>
std::shared_ptr<VectorType>
SparseMatrixDeviceOperator<VectorType>::build_range_vector() const
{
  auto partitioner =
      std::make_shared<const dealii::Utilities::MPI::Partitioner>(
          _matrix->locally_owned_range_indices(),
          _matrix->get_mpi_communicator());

  return std::make_shared<vector_type>(partitioner);
}

//-------------------------------------------------------------------------//

namespace internal
{
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
}

template <typename VectorType>
SmootherDeviceOperator<VectorType>::SmootherDeviceOperator(
    matrix_type const &matrix,
    std::shared_ptr<boost::property_tree::ptree> params)
    : _matrix(matrix)
{
  std::string prec_type = params->get("smoother.type", "Jacobi");
  initialize(prec_type);
}

template <typename VectorType>
void SmootherDeviceOperator<VectorType>::apply(VectorType const &b,
                                               VectorType &x) const
{
  // r = -(b - Ax)
  vector_type r(b);
  _matrix.vmult(r, x);
  r.add(-1., b);

  // x = x + B^{-1} (-r)
  vector_type tmp(x);
  _smoother.vmult(tmp, r);
  x.add(-1., tmp);
}

template <typename VectorType>
std::shared_ptr<VectorType>
SmootherDeviceOperator<VectorType>::build_domain_vector() const
{
  auto partitioner =
      std::make_shared<const dealii::Utilities::MPI::Partitioner>(
          _matrix.locally_owned_domain_indices(),
          _matrix.get_mpi_communicator());

  return std::make_shared<vector_type>(partitioner);
}

template <typename VectorType>
std::shared_ptr<VectorType>
SmootherDeviceOperator<VectorType>::build_range_vector() const
{
  auto partitioner =
      std::make_shared<const dealii::Utilities::MPI::Partitioner>(
          _matrix.locally_owned_range_indices(),
          _matrix.get_mpi_communicator());

  return std::make_shared<vector_type>(partitioner);
}

template <typename VectorType>
void SmootherDeviceOperator<VectorType>::initialize(std::string &prec_type)
{
  // Transform to lower case
  std::transform(prec_type.begin(), prec_type.end(), prec_type.begin(),
                 tolower);

  ASSERT_THROW(prec_type == "jacobi", "Only Jacobi smoother is implemented.");

  ASSERT(_matrix.m() == _matrix.n(),
         "The matrix is not square. The matrix is a " +
             std::to_string(_matrix.m()) + " by " +
             std::to_string(_matrix.n()) + " .");

  // Extract diagonal elements
  unsigned int const size = _matrix.n_local_rows();
  value_type *val_dev = nullptr;
  cuda_malloc(val_dev, size);
  int *column_index_dev = nullptr;
  cuda_malloc(column_index_dev, size);
  int *row_ptr_dev = nullptr;
  cuda_malloc(row_ptr_dev, size + 1);
  unsigned int const local_nnz = _matrix.local_nnz();
  int *row_index_coo_dev = nullptr;
  cuda_malloc(row_index_coo_dev, local_nnz);

  // Change to COO format. The only thing that needs to be change to go from CSR
  // to COO is to change row_ptr_dev with row_index_coo_dev.
  cusparseStatus_t cusparse_error_code =
      cusparseXcsr2coo(_matrix.cusparse_handle, _matrix.row_ptr_dev, local_nnz,
                       size, row_index_coo_dev, CUSPARSE_INDEX_BASE_ZERO);
  ASSERT_CUSPARSE(cusparse_error_code);

  int n_blocks = 1 + (local_nnz - 1) / BLOCK_SIZE;
  internal::extract_inv_diag<<<n_blocks, BLOCK_SIZE>>>(
      _matrix.val_dev, _matrix.column_index_dev, row_index_coo_dev, local_nnz,
      val_dev);

  iota<<<n_blocks, BLOCK_SIZE>>>(size, column_index_dev);

  n_blocks = 1 + size / BLOCK_SIZE;
  iota<<<n_blocks, BLOCK_SIZE>>>(size + 1, row_ptr_dev);

  _smoother.reinit(_matrix.get_mpi_communicator(), val_dev, column_index_dev,
                   row_ptr_dev, size, _matrix.locally_owned_range_indices(),
                   _matrix.locally_owned_range_indices(),
                   _matrix.cusparse_handle);

  cuda_free(row_index_coo_dev);
}

//-------------------------------------------------------------------------//

template <typename VectorType>
DirectDeviceOperator<VectorType>::DirectDeviceOperator(
    cusolverDnHandle_t cusolver_dn_handle,
    cusolverSpHandle_t cusolver_sp_handle,
    SparseMatrixDevice<typename VectorType::value_type> const &matrix,
    std::string const &solver)
    : _cusolver_dn_handle(cusolver_dn_handle),
      _cusolver_sp_handle(cusolver_sp_handle), _matrix(matrix), _solver(solver)
{
  // Transform to lower case
  std::transform(_solver.begin(), _solver.end(), _solver.begin(), tolower);
}

template <typename VectorType>
void DirectDeviceOperator<VectorType>::apply(VectorType const &b,
                                             VectorType &x) const
{
  if (_solver == "cholesky")
    cholesky_factorization(_cusolver_sp_handle, _matrix, b.get_values(),
                           x.get_values());
  else if (_solver == "lu_dense")
    lu_factorization(_cusolver_dn_handle, _matrix, b.get_values(),
                     x.get_values());
  else if (_solver == "lu_sparse_host")
    lu_factorization(_cusolver_sp_handle, _matrix, b.get_values(),
                     x.get_values());
  else
    ASSERT_THROW(false, "The provided solver name " + _solver + " is invalid.");
}

template <typename VectorType>
std::shared_ptr<VectorType>
DirectDeviceOperator<VectorType>::build_domain_vector() const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
}

template <typename VectorType>
std::shared_ptr<VectorType>
DirectDeviceOperator<VectorType>::build_range_vector() const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
}
}

#endif
