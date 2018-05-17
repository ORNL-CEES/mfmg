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

#include <deal.II/base/mpi.h>

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
           cusparseMatDescr_t const descr, float const *A_val_dev,
           int const *A_row_ptr_dev, int const *A_column_index_dev,
           float const *x, bool add, float *y)
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
           cusparseMatDescr_t const descr, double const *A_val_dev,
           int const *A_row_ptr_dev, int const *A_column_index_dev,
           double const *x, bool add, double *y)
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

void csrgemm(SparseMatrixDevice<float> const &A,
             SparseMatrixDevice<float> const &B, SparseMatrixDevice<float> &C)
{
  cusparseStatus_t error_code;
  cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_NON_TRANSPOSE;

  // This function performs C = op(A)*op(B)
  error_code = cusparseScsrgemm(
      A.cusparse_handle, cusparse_operation, cusparse_operation,
      A.n_local_rows(), B.n(), A.n(), A.descr, A.local_nnz(), A.val_dev,
      A.row_ptr_dev, A.column_index_dev, B.descr, B.local_nnz(), B.val_dev,
      B.row_ptr_dev, B.column_index_dev, C.descr, C.val_dev, C.row_ptr_dev,
      C.column_index_dev);
  ASSERT_CUSPARSE(error_code);
}

void csrgemm(SparseMatrixDevice<double> const &A,
             SparseMatrixDevice<double> const &B, SparseMatrixDevice<double> &C)
{
  cusparseStatus_t error_code;
  cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_NON_TRANSPOSE;

  // This function performs C = op(A)*op(B)
  error_code = cusparseDcsrgemm(
      A.cusparse_handle, cusparse_operation, cusparse_operation,
      A.n_local_rows(), B.n(), A.n(), A.descr, A.local_nnz(), A.val_dev,
      A.row_ptr_dev, A.column_index_dev, B.descr, B.local_nnz(), B.val_dev,
      B.row_ptr_dev, B.column_index_dev, C.descr, C.val_dev, C.row_ptr_dev,
      C.column_index_dev);
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

  int n_blocks = 1 + (size - 1) / block_size;
  internal::reorder_vector<<<n_blocks, block_size>>>(size, dst_buffer,
                                                     indices_dev, dst);
  cuda_free(dst_buffer);
}
}

template <typename ScalarType>
SparseMatrixDevice<ScalarType>::SparseMatrixDevice()
    : _comm(MPI_COMM_SELF), val_dev(nullptr), column_index_dev(nullptr),
      row_ptr_dev(nullptr), cusparse_handle(nullptr), descr(nullptr)
{
}

template <typename ScalarType>
SparseMatrixDevice<ScalarType>::SparseMatrixDevice(
    SparseMatrixDevice<ScalarType> &&other)
    : _comm(other._comm), val_dev(other.val_dev),
      column_index_dev(other.column_index_dev), row_ptr_dev(other.row_ptr_dev),
      cusparse_handle(other.cusparse_handle), descr(other.descr),
      _local_nnz(other._local_nnz), _nnz(other._nnz),
      _range_indexset(other._range_indexset),
      _domain_indexset(other._domain_indexset)
{
  other.val_dev = nullptr;
  other.column_index_dev = nullptr;
  other.row_ptr_dev = nullptr;
  other.cusparse_handle = nullptr;
  other.descr = nullptr;

  other._local_nnz = 0;
  other._nnz = 0;
  other._range_indexset.clear();
  other._domain_indexset.clear();
}

template <typename ScalarType>
SparseMatrixDevice<ScalarType>::SparseMatrixDevice(
    SparseMatrixDevice<ScalarType> const &other)
    : _comm(other._comm), cusparse_handle(other.cusparse_handle),
      _local_nnz(other._local_nnz), _nnz(other._nnz),
      _range_indexset(other._range_indexset),
      _domain_indexset(other._domain_indexset)
{
  cuda_malloc(val_dev, _local_nnz);
  cudaError_t cuda_error_code;
  cuda_error_code =
      cudaMemcpy(val_dev, other.val_dev, _local_nnz * sizeof(ScalarType),
                 cudaMemcpyDeviceToDevice);
  ASSERT_CUDA(cuda_error_code);

  cuda_malloc(column_index_dev, _local_nnz);
  cuda_error_code =
      cudaMemcpy(column_index_dev, other.column_index_dev,
                 _local_nnz * sizeof(int), cudaMemcpyDeviceToDevice);
  ASSERT_CUDA(cuda_error_code);

  unsigned int const size = _range_indexset.n_elements() + 1;
  cuda_malloc(row_ptr_dev, size);
  cuda_error_code = cudaMemcpy(row_ptr_dev, other.row_ptr_dev,
                               size * sizeof(int), cudaMemcpyDeviceToDevice);
  ASSERT_CUDA(cuda_error_code);

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
SparseMatrixDevice<ScalarType>::SparseMatrixDevice(
    MPI_Comm comm, ScalarType *val_dev_, int *column_index_dev_,
    int *row_ptr_dev_, unsigned int local_nnz,
    dealii::IndexSet const &range_indexset,
    dealii::IndexSet const &domain_indexset)
    : _comm(comm), val_dev(val_dev_), column_index_dev(column_index_dev_),
      row_ptr_dev(row_ptr_dev_), cusparse_handle(nullptr), descr(nullptr),
      _local_nnz(local_nnz), _range_indexset(range_indexset),
      _domain_indexset(domain_indexset)
{
  _nnz = _local_nnz;
  dealii::Utilities::MPI::sum(_nnz, _comm);
}

template <typename ScalarType>
SparseMatrixDevice<ScalarType>::SparseMatrixDevice(
    MPI_Comm comm, ScalarType *val_dev_, int *column_index_dev_,
    int *row_ptr_dev_, unsigned int local_nnz,
    dealii::IndexSet const &range_indexset,
    dealii::IndexSet const &domain_indexset, cusparseHandle_t handle)
    : SparseMatrixDevice<ScalarType>(comm, val_dev_, column_index_dev_,
                                     row_ptr_dev_, local_nnz, range_indexset,
                                     domain_indexset)
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
  {
    cuda_free(val_dev);
    val_dev = nullptr;
  }

  if (column_index_dev != nullptr)
  {
    cuda_free(column_index_dev);
    column_index_dev = nullptr;
  }

  if (row_ptr_dev != nullptr)
  {
    cuda_free(row_ptr_dev);
    row_ptr_dev = nullptr;
  }

  if (descr != nullptr)
  {
    cusparseStatus_t cusparse_error_code;
    cusparse_error_code = cusparseDestroyMatDescr(descr);
    ASSERT_CUSPARSE(cusparse_error_code);
    descr = nullptr;
  }
}

template <typename ScalarType>
SparseMatrixDevice<ScalarType> &SparseMatrixDevice<ScalarType>::
operator=(SparseMatrixDevice<ScalarType> &&other)
{
  _comm = other._comm;
  val_dev = other.val_dev;
  column_index_dev = other.column_index_dev;
  row_ptr_dev = other.row_ptr_dev;
  cusparse_handle = other.cusparse_handle;
  descr = other.descr;
  _local_nnz = other._local_nnz;
  _nnz = other._nnz;
  _range_indexset = other._range_indexset;
  _domain_indexset = other._domain_indexset;

  other.val_dev = nullptr;
  other.column_index_dev = nullptr;
  other.row_ptr_dev = nullptr;
  other.cusparse_handle = nullptr;
  other.descr = nullptr;

  other._local_nnz = 0;
  other._nnz = 0;
  other._range_indexset.clear();
  other._domain_indexset.clear();

  return *this;
}

template <typename ScalarType>
void SparseMatrixDevice<ScalarType>::reinit(
    MPI_Comm comm, ScalarType *val_dev_, int *column_index_dev_,
    int *row_ptr_dev_, unsigned int local_nnz,
    dealii::IndexSet const &range_indexset,
    dealii::IndexSet const &domain_indexset, cusparseHandle_t cusparse_handle_)
{
  // This function can only be called if the object is empty. Otherwise we need
  // to deal with the ownership of the data
  ASSERT((val_dev == nullptr) && (column_index_dev == nullptr) &&
             (row_ptr_dev == nullptr),
         "Cannot reinit a SparseMatrixDevice which is not empty.");
  val_dev = val_dev_;
  column_index_dev = column_index_dev_;
  row_ptr_dev = row_ptr_dev_;

  cusparse_handle = cusparse_handle_;
  cusparseStatus_t cusparse_error_code;
  cusparse_error_code = cusparseCreateMatDescr(&descr);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code = cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code =
      cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  ASSERT_CUSPARSE(cusparse_error_code);

  _comm = comm;
  _local_nnz = local_nnz;
  _nnz = _local_nnz;
  dealii::Utilities::MPI::sum(_nnz, _comm);
  _range_indexset = range_indexset;
  _domain_indexset = domain_indexset;
}

template <typename ScalarType>
dealii::IndexSet
SparseMatrixDevice<ScalarType>::locally_owned_domain_indices() const
{
  return _domain_indexset;
}

template <typename ScalarType>
dealii::IndexSet
SparseMatrixDevice<ScalarType>::locally_owned_range_indices() const
{
  return _range_indexset;
}

template <typename ScalarType>
void SparseMatrixDevice<ScalarType>::vmult(
    VectorDevice<ScalarType> &dst, VectorDevice<ScalarType> const &src) const
{
  // Get the whole src vector
  unsigned int const size = src.partitioner->size();
  ScalarType *src_val_dev;
  cudaError_t cuda_error_code;
  cuda_error_code = cudaMalloc(&src_val_dev, size * sizeof(ScalarType));
  ASSERT_CUDA(cuda_error_code);
  internal::gather_vector<ScalarType>(src, src_val_dev);

  // Perform the matrix-vector multiplication on the row that are locally
  // owned
  internal::csrmv(cusparse_handle, false, _range_indexset.n_elements(), size,
                  _local_nnz, descr, val_dev, row_ptr_dev, column_index_dev,
                  src_val_dev, false, dst.val_dev);
}

template <typename ScalarType>
void SparseMatrixDevice<ScalarType>::mmult(
    SparseMatrixDevice<ScalarType> &C,
    SparseMatrixDevice<ScalarType> const &B) const
{
  // Compute the number of non-zero elements in C
  ASSERT(B.m() == n(), "The matrices cannot be multiplied together. You are "
                       "trying to mutiply a " +
                           std::to_string(m()) + " by " + std::to_string(n()) +
                           " matrix with a " + std::to_string(B.m()) + " by " +
                           std::to_string(B.n()));

  unsigned int const comm_size = dealii::Utilities::MPI::n_mpi_processes(_comm);

  // If the code is serial then we can use cusparse directly. Otherwise, we need
  // to use dealii.
  if (comm_size == 1)
  {
    // Reinitialize part of C
    cuda_free(C.row_ptr_dev);
    cuda_malloc(C.row_ptr_dev, n_local_rows() + 1);

    int C_local_nnz = 0;
    cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseStatus_t cusparse_error_code;
    cusparse_error_code = cusparseXcsrgemmNnz(
        cusparse_handle, cusparse_operation, cusparse_operation, n_local_rows(),
        B.n(), n(), descr, local_nnz(), row_ptr_dev, column_index_dev, B.descr,
        B.local_nnz(), B.row_ptr_dev, B.column_index_dev, C.descr,
        C.row_ptr_dev, &C_local_nnz);
    ASSERT_CUSPARSE(cusparse_error_code);

    // Reinitialize part of C
    cuda_free(C.val_dev);
    cuda_free(C.column_index_dev);
    cuda_malloc(C.val_dev, C_local_nnz);
    cuda_malloc(C.column_index_dev, C_local_nnz);
    C._local_nnz = C_local_nnz;
    C._nnz = C._local_nnz;
    C._range_indexset = _range_indexset;
    C._domain_indexset = B._domain_indexset;

    internal::csrgemm(*this, B, C);
  }
  else
  {
    dealii::TrilinosWrappers::SparseMatrix C_host;
    auto A_host = convert_to_trilinos_matrix(*this);
    auto B_host = convert_to_trilinos_matrix(B);

    A_host.mmult(C_host, B_host);

    C = convert_matrix(C_host);
  }
}
}

#endif
