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

#include <mfmg/common/exceptions.hpp>
#include <mfmg/cuda/dealii_operator_device_helpers.cuh>
#include <mfmg/cuda/utils.cuh>

namespace mfmg
{
namespace
{
void cusparsecsr2dense(SparseMatrixDevice<float> const &matrix,
                       float *dense_matrix_dev)
{
  cusparseStatus_t cusparse_error_code;
  cusparse_error_code =
      cusparseScsr2dense(matrix.cusparse_handle, matrix.m(), matrix.n(),
                         matrix.descr, matrix.val_dev, matrix.row_ptr_dev,
                         matrix.column_index_dev, dense_matrix_dev, matrix.m());
  ASSERT_CUSPARSE(cusparse_error_code);
}

void cusparsecsr2dense(SparseMatrixDevice<double> const &matrix,
                       double *dense_matrix_dev)
{
  cusparseStatus_t cusparse_error_code;
  cusparse_error_code =
      cusparseDcsr2dense(matrix.cusparse_handle, matrix.m(), matrix.n(),
                         matrix.descr, matrix.val_dev, matrix.row_ptr_dev,
                         matrix.column_index_dev, dense_matrix_dev, matrix.m());
  ASSERT_CUSPARSE(cusparse_error_code);
}

void cusolverDngetrf_buffer_size(cusolverDnHandle_t cusolver_dn_handle, int m,
                                 int n, float *dense_matrix_dev,
                                 int &workspace_size)
{
  cusolverStatus_t cusolver_error_code;
  cusolver_error_code = cusolverDnSgetrf_bufferSize(
      cusolver_dn_handle, m, n, dense_matrix_dev, m, &workspace_size);
  ASSERT_CUSOLVER(cusolver_error_code);
}

void cusolverDngetrf_buffer_size(cusolverDnHandle_t cusolver_dn_handle, int m,
                                 int n, double *dense_matrix_dev,
                                 int &workspace_size)
{
  cusolverStatus_t cusolver_error_code;
  cusolver_error_code = cusolverDnDgetrf_bufferSize(
      cusolver_dn_handle, m, n, dense_matrix_dev, m, &workspace_size);
  ASSERT_CUSOLVER(cusolver_error_code);
}

void cusolverDngetrf(cusolverDnHandle_t cusolver_dn_handle, int m, int n,
                     float *dense_matrix_dev, float *workspace_dev,
                     int *pivot_dev, int *info_dev)
{
  cusolverStatus_t cusolver_error_code;
  cusolver_error_code =
      cusolverDnSgetrf(cusolver_dn_handle, m, n, dense_matrix_dev, m,
                       workspace_dev, pivot_dev, info_dev);
  ASSERT_CUSOLVER(cusolver_error_code);
}

void cusolverDngetrf(cusolverDnHandle_t cusolver_dn_handle, int m, int n,
                     double *dense_matrix_dev, double *workspace_dev,
                     int *pivot_dev, int *info_dev)
{
  cusolverStatus_t cusolver_error_code;
  cusolver_error_code =
      cusolverDnDgetrf(cusolver_dn_handle, m, n, dense_matrix_dev, m,
                       workspace_dev, pivot_dev, info_dev);
  ASSERT_CUSOLVER(cusolver_error_code);
}

void cusolverDngetrs(cusolverDnHandle_t cusolver_dn_handle, int m,
                     float *dense_matrix_dev, int *pivot_dev, float *b,
                     int *info_dev)
{
  int const n_rhs = 1;
  cusolverStatus_t cusolver_error_code;
  cusolver_error_code =
      cusolverDnSgetrs(cusolver_dn_handle, CUBLAS_OP_N, m, n_rhs,
                       dense_matrix_dev, m, pivot_dev, b, m, info_dev);
  ASSERT_CUSOLVER(cusolver_error_code);
}

void cusolverDngetrs(cusolverDnHandle_t cusolver_dn_handle, int m,
                     double *dense_matrix_dev, int *pivot_dev, double *b,
                     int *info_dev)
{
  int const n_rhs = 1;
  cusolverStatus_t cusolver_error_code;
  cusolver_error_code =
      cusolverDnDgetrs(cusolver_dn_handle, CUBLAS_OP_N, m, n_rhs,
                       dense_matrix_dev, m, pivot_dev, b, m, info_dev);
  ASSERT_CUSOLVER(cusolver_error_code);
}

void cusolverSpcsrlsvluHost(cusolverSpHandle_t cusolver_sp_handle,
                            unsigned int const n_rows, unsigned int const nnz,
                            cusparseMatDescr_t descr, float const *val_host,
                            int const *row_ptr_host,
                            int const *column_index_host, float const *b_host,
                            float *x_host)
{
  int singularity = 0;
  cusolverStatus_t cusolver_error_code = cusolverSpScsrlsvluHost(
      cusolver_sp_handle, n_rows, nnz, descr, val_host, row_ptr_host,
      column_index_host, b_host, 0., 1, x_host, &singularity);
  ASSERT_CUSOLVER(cusolver_error_code);
  ASSERT(singularity == -1, "Coarse matrix is singular");
}

void cusolverSpcsrlsvluHost(cusolverSpHandle_t cusolver_sp_handle,
                            unsigned int const n_rows, unsigned int nnz,
                            cusparseMatDescr_t descr, double const *val_host,
                            int const *row_ptr_host,
                            int const *column_index_host, double const *b_host,
                            double *x_host)
{
  int singularity = 0;
  cusolverStatus_t cusolver_error_code = cusolverSpDcsrlsvluHost(
      cusolver_sp_handle, n_rows, nnz, descr, val_host, row_ptr_host,
      column_index_host, b_host, 0., 1, x_host, &singularity);
  ASSERT_CUSOLVER(cusolver_error_code);
  ASSERT(singularity == -1, "Coarse matrix is singular");
}
} // namespace

void cholesky_factorization(cusolverSpHandle_t cusolver_sp_handle,
                            SparseMatrixDevice<float> const &matrix,
                            float const *b, float *x)
{
  int singularity = 0;

  cusolverStatus_t cusolver_error_code = cusolverSpScsrlsvchol(
      cusolver_sp_handle, matrix.m(), matrix.n_nonzero_elements(), matrix.descr,
      matrix.val_dev, matrix.row_ptr_dev, matrix.column_index_dev, b, 0., 0, x,
      &singularity);
  ASSERT_CUSOLVER(cusolver_error_code);

  ASSERT(singularity == -1, "Coarse matrix is not SPD");
}

void cholesky_factorization(cusolverSpHandle_t cusolver_sp_handle,
                            SparseMatrixDevice<double> const &matrix,
                            double const *b, double *x)
{
  int singularity = 0;

  cusolverStatus_t cusolver_error_code = cusolverSpDcsrlsvchol(
      cusolver_sp_handle, matrix.m(), matrix.n_nonzero_elements(), matrix.descr,
      matrix.val_dev, matrix.row_ptr_dev, matrix.column_index_dev, b, 0., 0, x,
      &singularity);
  ASSERT_CUSOLVER(cusolver_error_code);

  ASSERT(singularity == -1, "Coarse matrix is not SPD");
}

template <typename ScalarType>
void lu_factorization(cusolverDnHandle_t cusolver_dn_handle,
                      SparseMatrixDevice<ScalarType> const &matrix,
                      ScalarType const *b_dev, ScalarType *x_dev)
{
  // Change the format of the matrix from sparse to dense
  unsigned int const m = matrix.m();
  unsigned int const n = matrix.n();
  ASSERT(m == n, "The matrix is not square");
  ScalarType *dense_matrix_dev;
  cuda_malloc(dense_matrix_dev, m * n);

  // Change the format of matrix to dense
  cusparsecsr2dense(matrix, dense_matrix_dev);

  // Create the working space
  int workspace_size = 0;
  cusolverDngetrf_buffer_size(cusolver_dn_handle, m, n, dense_matrix_dev,
                              workspace_size);
  ASSERT(workspace_size > 0, "No workspace was allocated");
  ScalarType *workspace_dev;
  cuda_malloc(workspace_dev, workspace_size);

  // LU factorization
  int *pivot_dev;
  cuda_malloc(pivot_dev, m);
  int *info_dev;
  cuda_malloc(info_dev, 1);

  cusolverDngetrf(cusolver_dn_handle, m, n, dense_matrix_dev, workspace_dev,
                  pivot_dev, info_dev);

  cudaError_t cuda_error_code;
#ifdef MFMG_DEBUG
  int info = 0;
  cuda_error_code =
      cudaMemcpy(&info, info_dev, sizeof(int), cudaMemcpyDeviceToHost);
  ASSERT_CUDA(cuda_error_code);
  ASSERT(info == 0, "There was a problem during the LU factorization");
#endif

  // Solve Ax = b
  cuda_error_code = cudaMemcpy(x_dev, b_dev, m * sizeof(ScalarType),
                               cudaMemcpyDeviceToDevice);
  ASSERT_CUDA(cuda_error_code);
  cusolverDngetrs(cusolver_dn_handle, m, dense_matrix_dev, pivot_dev, x_dev,
                  info_dev);
#ifdef MFMG_DEBUG
  cuda_error_code =
      cudaMemcpy(&info, info_dev, sizeof(int), cudaMemcpyDeviceToHost);
  ASSERT_CUDA(cuda_error_code);
  ASSERT(info == 0, "There was a problem during the LU solve");
#endif

  // Free the memory allocated
  cuda_free(dense_matrix_dev);
  cuda_free(workspace_dev);
  cuda_free(pivot_dev);
  cuda_free(info_dev);
}

template <typename ScalarType>
void lu_factorization(cusolverSpHandle_t cusolver_sp_handle,
                      SparseMatrixDevice<ScalarType> const &matrix,
                      ScalarType const *b_dev, ScalarType *x_dev)
{
  // cuSOLVER does not support LU factorization of sparse matrix on the device,
  // so we need to move everything to the host first and then back to the host.
  unsigned int const nnz = matrix.n_nonzero_elements();
  unsigned int const n_rows = matrix.m();
  std::vector<ScalarType> val_host(nnz);
  std::vector<int> column_index_host(nnz);
  std::vector<int> row_ptr_host(n_rows + 1);
  cuda_mem_copy_to_host(matrix.val_dev, val_host);
  cuda_mem_copy_to_host(matrix.column_index_dev, column_index_host);
  cuda_mem_copy_to_host(matrix.row_ptr_dev, row_ptr_host);
  std::vector<ScalarType> b_host(n_rows);
  cuda_mem_copy_to_host(b_dev, b_host);
  std::vector<ScalarType> x_host(n_rows);
  cuda_mem_copy_to_host(x_dev, x_host);

  cusolverSpcsrlsvluHost(cusolver_sp_handle, n_rows, nnz, matrix.descr,
                         val_host.data(), row_ptr_host.data(),
                         column_index_host.data(), b_host.data(),
                         x_host.data());

  // Move the solution back to the device
  cuda_mem_copy_to_dev(x_host, x_dev);
}

__global__ void iota(int const size, int *value)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
    value[idx] = idx;
}

template void lu_factorization<float>(cusolverDnHandle_t cusolver_dn_handle,
                                      SparseMatrixDevice<float> const &matrix,
                                      float const *b, float *x);
template void lu_factorization<double>(cusolverDnHandle_t cusolver_dn_handle,
                                       SparseMatrixDevice<double> const &matrix,
                                       double const *b, double *x);

template void lu_factorization<float>(cusolverSpHandle_t cusolver_sp_handle,
                                      SparseMatrixDevice<float> const &matrix,
                                      float const *b, float *x);
template void lu_factorization<double>(cusolverSpHandle_t cusolver_sp_handle,
                                       SparseMatrixDevice<double> const &matrix,
                                       double const *b, double *x);
} // namespace mfmg
