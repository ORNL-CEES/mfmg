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

#ifndef SPARSE_MATRIX_DEVICE_TEMPLATES_CUH
#define SPARSE_MATRIX_DEVICE_TEMPLATES_CUH

#include <mfmg/exceptions.hpp>
#include <mfmg/sparse_matrix_device.cuh>
#include <mfmg/utils.cuh>

#define BLOCK_SIZE 512

namespace mfmg
{
namespace internal
{
template <typename ScalarType>
__global__ void reorder_vector(int const size, ScalarType *src,
                               unsigned int *indices, ScalarType *dst)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size)
    dst[indices[i]] = src[i];
}

void csrmv(cusparseHandle_t handle, bool transpose, int m, int n, int nnz,
           const cusparseMatDescr_t descr, const float *A_val_dev,
           const int *A_row_ptr_dev, const int *A_column_index_dev,
           const float *x, bool add, float *y)
{
  float alpha = 1.;
  float beta = add ? 1. : 0.;
  cusparseOperation_t cusparse_operation =
      transpose ? CUSPARSE_OPERATION_TRANSPOSE
                : CUSPARSE_OPERATION_NON_TRANSPOSE;

  cusparseStatus_t error_code;
  // This function performs y = alpha*op(A)*x + beta*y
  error_code =
      cusparseScsrmv(handle, cusparse_operation, m, n, nnz, &alpha, descr,
                     A_val_dev, A_row_ptr_dev, A_column_index_dev, x, &beta, y);
  ASSERT_CUSPARSE(error_code);
}

void csrmv(cusparseHandle_t handle, bool transpose, int m, int n, int nnz,
           const cusparseMatDescr_t descr, const double *A_val_dev,
           const int *A_row_ptr_dev, const int *A_column_index_dev,
           const double *x, bool add, double *y)
{
  double alpha = 1.;
  double beta = add ? 1. : 0.;
  cusparseOperation_t cusparse_operation =
      transpose ? CUSPARSE_OPERATION_TRANSPOSE
                : CUSPARSE_OPERATION_NON_TRANSPOSE;

  cusparseStatus_t error_code;
  // This function performs y = alpha*op(A)*x + beta*y
  error_code =
      cusparseDcsrmv(handle, cusparse_operation, m, n, nnz, &alpha, descr,
                     A_val_dev, A_row_ptr_dev, A_column_index_dev, x, &beta, y);
  ASSERT_CUSPARSE(error_code);
}

template <typename ScalarType>
void gather_vector(VectorDevice<ScalarType> const &src, ScalarType *dst)
{
  MPI_Comm comm = src.partitioner->get_mpi_communicator();
  unsigned int const local_size = src.partitioner->local_size();
  unsigned int const size = src.partitioner->size();

  // Because each processor needs to know the full vector, we can do an
  // all-gather communication
  ScalarType *dst_buffer;
  cuda_malloc(dst_buffer, size);
  all_gather_dev(comm, local_size, src.val_dev, size, dst_buffer);

  // All-gather on the indices
  std::vector<unsigned int> local_indices;
  src.partitioner->locally_owned_range().fill_index_vector(local_indices);
  std::vector<unsigned int> indices(size);
  all_gather(comm, local_size, local_indices.data(), size, indices.data());

  // Reorder the elements in the dst array
  unsigned int *indices_dev;
  cuda_malloc(indices_dev, size);
  cudaError_t cuda_error_code;
  cuda_error_code =
      cudaMemcpy(indices_dev, indices.data(), size * sizeof(unsigned int),
                 cudaMemcpyHostToDevice);
  ASSERT_CUDA(cuda_error_code);

  int n_blocks = 1 + (size - 1) / BLOCK_SIZE;
  internal::reorder_vector<<<n_blocks, BLOCK_SIZE>>>(size, dst_buffer,
                                                     indices_dev, dst);
  cuda_free(dst_buffer);
}
}

template <typename ScalarType>
SparseMatrixDevice<ScalarType>::SparseMatrixDevice()
    : val_dev(nullptr), column_index_dev(nullptr), row_ptr_dev(nullptr),
      cusparse_handle(nullptr), descr(nullptr), nnz(0), n_rows(0)
{
}

template <typename ScalarType>
SparseMatrixDevice<ScalarType>::SparseMatrixDevice(
    SparseMatrixDevice<ScalarType> &&other)
    : val_dev(other.val_dev), column_index_dev(other.column_index_dev),
      row_ptr_dev(other.row_ptr_dev), cusparse_handle(other.cusparse_handle),
      descr(other.descr), nnz(other.nnz), n_rows(other.n_rows)
{
  other.val_dev = nullptr;
  other.column_index_dev = nullptr;
  other.row_ptr_dev = nullptr;
  other.cusparse_handle = nullptr;
  other.descr = nullptr;

  other.nnz = 0;
  other.n_rows = 0;
}

template <typename ScalarType>
SparseMatrixDevice<ScalarType>::SparseMatrixDevice(ScalarType *val_dev_,
                                                   int *column_index_dev_,
                                                   int *row_ptr_dev_,
                                                   unsigned int nnz_,
                                                   unsigned int n_rows_)
    : val_dev(val_dev_), column_index_dev(column_index_dev_),
      row_ptr_dev(row_ptr_dev_), cusparse_handle(nullptr), descr(nullptr),
      nnz(nnz_), n_rows(n_rows_)
{
}

template <typename ScalarType>
SparseMatrixDevice<ScalarType>::SparseMatrixDevice(
    ScalarType *val_dev_, int *column_index_dev_, int *row_ptr_dev_,
    cusparseHandle_t handle, unsigned int nnz_, unsigned int n_rows_)
    : SparseMatrixDevice<ScalarType>(val_dev_, column_index_dev_, row_ptr_dev_,
                                     nnz_, n_rows_)
{
  cusparse_handle = handle;

  cusparseStatus_t cusparse_error_code;
  cusparse_error_code = cusparseCreateMatDescr(&descr);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code = cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code =
      cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  ASSERT_CUSPARSE(cusparse_error_code);
}

template <typename ScalarType>
SparseMatrixDevice<ScalarType>::~SparseMatrixDevice()
{
  if (val_dev != nullptr)
    cuda_free(val_dev);

  if (column_index_dev != nullptr)
    cuda_free(column_index_dev);

  if (row_ptr_dev != nullptr)
    cuda_free(row_ptr_dev);

  if (descr != nullptr)
  {
    cusparseStatus_t cusparse_error_code;
    cusparse_error_code = cusparseDestroyMatDescr(descr);
    ASSERT_CUSPARSE(cusparse_error_code);
  }
}
}

#endif
