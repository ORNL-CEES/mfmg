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

#ifndef AMGE_DEVICE_TEMPLATES_CUH
#define AMGE_DEVICE_TEMPLATES_CUH

#include <mfmg/amge_device.cuh>

#include <mfmg/utils.cuh>

#include <deal.II/dofs/dof_accessor.h>

#define BLOCK_SIZE 512

namespace mfmg
{
namespace internal
{
template <typename ScalarType>
void convert_csr_to_dense(cusparseHandle_t, cusparseMatDescr_t const,
                          std::shared_ptr<SparseMatrixDevice<ScalarType>> const,
                          ScalarType *&)
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <>
void convert_csr_to_dense<float>(
    cusparseHandle_t handle, cusparseMatDescr_t const descr,
    std::shared_ptr<SparseMatrixDevice<float>> const sparse_matrix_dev,
    float *&dense_matrix_dev)
{
  int n_rows = sparse_matrix_dev->n_rows;

  cudaError_t cuda_error_code;
  cuda_error_code =
      cudaMalloc(&dense_matrix_dev, n_rows * n_rows * sizeof(float));
  ASSERT_CUDA(cuda_error_code);

  cusparseStatus_t cusparse_error_code;
  cusparse_error_code = cusparseScsr2dense(
      handle, n_rows, n_rows, descr, sparse_matrix_dev->val_dev,
      sparse_matrix_dev->row_ptr_dev, sparse_matrix_dev->column_index_dev,
      dense_matrix_dev, n_rows);
  ASSERT_CUSPARSE(cusparse_error_code);
}

template <>
void convert_csr_to_dense<double>(
    cusparseHandle_t handle, cusparseMatDescr_t const descr,
    std::shared_ptr<SparseMatrixDevice<double>> const sparse_matrix_dev,
    double *&dense_matrix_dev)
{
  int n_rows = sparse_matrix_dev->n_rows;

  cudaError_t cuda_error_code;
  cuda_error_code =
      cudaMalloc(&dense_matrix_dev, n_rows * n_rows * sizeof(double));
  ASSERT_CUDA(cuda_error_code);

  cusparseStatus_t cusparse_error_code;
  cusparse_error_code = cusparseDcsr2dense(
      handle, n_rows, n_rows, descr, sparse_matrix_dev->val_dev,
      sparse_matrix_dev->row_ptr_dev, sparse_matrix_dev->column_index_dev,
      dense_matrix_dev, n_rows);
  ASSERT_CUSPARSE(cusparse_error_code);
}

template <typename ScalarType>
void compute_local_eigenvectors(cusolverDnHandle_t, int, ScalarType *,
                                ScalarType *, ScalarType *)
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <>
void compute_local_eigenvectors<float>(cusolverDnHandle_t handle, int n,
                                       float *A, float *B, float *W)
{
  // See https://docs.nvidia.com/cuda/cusolver/index.html#sygvd-example1

  // Query working space of sygvd
  cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  cusolverStatus_t cusolver_error_code;
  int lwork = 0;
  cusolver_error_code = cusolverDnSsygvd_bufferSize(handle, itype, jobz, uplo,
                                                    n, A, n, B, n, W, &lwork);
  ASSERT_CUSOLVER(cusolver_error_code);

  // Compute the spectrum. After the call the content of A is overwritten by the
  // orthonormal eigenvectors.
  float *d_work;
  cudaError_t cuda_error_code;
  cuda_error_code = cudaMalloc(&d_work, lwork * sizeof(float));
  ASSERT_CUDA(cuda_error_code);
  int *devInfo;
  cuda_error_code = cudaMalloc(&devInfo, sizeof(int));
  ASSERT_CUDA(cuda_error_code);
  cusolver_error_code = cusolverDnSsygvd(handle, itype, jobz, uplo, n, A, n, B,
                                         n, W, d_work, lwork, devInfo);
  ASSERT_CUSOLVER(cusolver_error_code);
#if MFMG_DEBUG
  int info_gpu = 0;
  cuda_error_code =
      cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  ASSERT_CUDA(cuda_error_code);
  ASSERT(info_gpu == 0, "sygvd error " + std::to_string(info_gpu));
#endif
  // Free memory
  cuda_free(d_work);
  cuda_free(devInfo);
}

template <>
void compute_local_eigenvectors<double>(cusolverDnHandle_t handle, int n,
                                        double *A, double *B, double *W)
{
  // See https://docs.nvidia.com/cuda/cusolver/index.html#sygvd-example1

  // Query working space of sygvd
  cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  cusolverStatus_t cusolver_error_code;
  int lwork = 0;
  cusolver_error_code = cusolverDnDsygvd_bufferSize(handle, itype, jobz, uplo,
                                                    n, A, n, B, n, W, &lwork);
  ASSERT_CUSOLVER(cusolver_error_code);

  std::vector<double> A_host(n * n);
  cudaMemcpy(&A_host[0], A, n * n * sizeof(double), cudaMemcpyDeviceToHost);

  // Compute the spectrum. After the call the content of A is overwritten by the
  // orthonormal eigenvectors.
  double *d_work;
  cudaError_t cuda_error_code;
  cuda_error_code = cudaMalloc(&d_work, lwork * sizeof(double));
  ASSERT_CUDA(cuda_error_code);
  int *devInfo;
  cuda_error_code = cudaMalloc(&devInfo, sizeof(int));
  ASSERT_CUDA(cuda_error_code);
  cusolver_error_code = cusolverDnDsygvd(handle, itype, jobz, uplo, n, A, n, B,
                                         n, W, d_work, lwork, devInfo);
  ASSERT_CUSOLVER(cusolver_error_code);
#if MFMG_DEBUG
  int info_gpu = 0;
  cuda_error_code =
      cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  ASSERT_CUDA(cuda_error_code);
  ASSERT(info_gpu == 0, "sygvd error " + std::to_string(info_gpu));
#endif
  // Free memory
  cuda_free(d_work);
  cuda_free(devInfo);
}

template <typename ScalarType>
__global__ void restrict_array(int full_array_size, ScalarType *full_array,
                               int restrict_array_size,
                               ScalarType *restricted_array)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < restrict_array_size)
    restricted_array[i] = full_array[i];
}
}

template <int dim, typename VectorType>
AMGe_device<dim, VectorType>::AMGe_device(
    MPI_Comm comm, dealii::DoFHandler<dim> const &dof_handler,
    cusolverDnHandle_t cusolver_dn_handle, cusparseHandle_t cusparse_handle)
    : AMGe<dim, VectorType>(comm, dof_handler),
      _cusolver_dn_handle(cusolver_dn_handle), _cusparse_handle(cusparse_handle)
{
}

// Cannot be const because of the handles
template <int dim, typename VectorType>
std::tuple<typename VectorType::value_type *, typename VectorType::value_type *,
           std::vector<dealii::types::global_dof_index>>
AMGe_device<dim, VectorType>::compute_local_eigenvectors(
    unsigned int n_eigenvectors,
    dealii::Triangulation<dim> const &agglomerate_triangulation,
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator> const
        &patch_to_global_map,
    std::function<void(dealii::DoFHandler<dim> &, dealii::ConstraintMatrix &,
                       std::shared_ptr<SparseMatrixDevice<ScalarType>> &,
                       std::shared_ptr<SparseMatrixDevice<ScalarType>> &)> const
        &evaluate)
{
  dealii::DoFHandler<dim> agglomerate_dof_handler(agglomerate_triangulation);
  // Not used for now. It will be used once we have an implementation of lanczos
  // algorithm
  dealii::ConstraintMatrix agglomerate_constraints;

  std::shared_ptr<SparseMatrixDevice<ScalarType>> system_matrix_dev;
  std::shared_ptr<SparseMatrixDevice<ScalarType>> mass_matrix_dev;

  // Call user function
  evaluate(agglomerate_dof_handler, agglomerate_constraints, system_matrix_dev,
           mass_matrix_dev);

  // Convert the matrix from CRS to dense. First, create and setup matrix
  // descriptor
  cusparseStatus_t cusparse_error_code;
  cusparseMatDescr_t descr;
  cusparse_error_code = cusparseCreateMatDescr(&descr);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code = cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code =
      cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  ASSERT_CUSPARSE(cusparse_error_code);
  int const n_rows = system_matrix_dev->n_rows;

  // Convert the system matrix to dense
  ScalarType *dense_system_matrix_dev = nullptr;
  internal::convert_csr_to_dense(_cusparse_handle, descr, system_matrix_dev,
                                 dense_system_matrix_dev);
  // Free the memory of the system sparse matrix
  system_matrix_dev.reset();

  // Convert the mass matrix to dense
  ScalarType *dense_mass_matrix_dev = nullptr;
  internal::convert_csr_to_dense(_cusparse_handle, descr, mass_matrix_dev,
                                 dense_mass_matrix_dev);
  // Free the memory of the mass sparse matrix
  cusparse_error_code = cusparseDestroyMatDescr(descr);
  ASSERT_CUSPARSE(cusparse_error_code);
  mass_matrix_dev.reset();

  // Compute the eigenvalues and the eigenvectors. The values in
  // dense_system_matrix_dev are overwritten and replaced by the eigenvectors
  ScalarType *eigenvalues_dev = nullptr;
  cudaError_t cuda_error_code;
  cuda_error_code = cudaMalloc(&eigenvalues_dev, n_rows * sizeof(ScalarType));
  internal::compute_local_eigenvectors(_cusolver_dn_handle, n_rows,
                                       dense_system_matrix_dev,
                                       dense_mass_matrix_dev, eigenvalues_dev);
  cuda_free(dense_mass_matrix_dev);
  // We now have too many eigenvectors. So we only keep the ones associated to
  // the smallest ones.
  ScalarType *smallest_eigenvalues_dev = nullptr;
  cuda_error_code = cudaMalloc(&smallest_eigenvalues_dev,
                               n_eigenvectors * sizeof(ScalarType));

  ASSERT_CUDA(cuda_error_code);
  int n_blocks = 1 + (n_eigenvectors - 1) / BLOCK_SIZE;
  internal::restrict_array<<<n_blocks, BLOCK_SIZE>>>(
      n_rows, eigenvalues_dev, n_eigenvectors, smallest_eigenvalues_dev);
  // Check that the kernel was launched correctly
  ASSERT_CUDA(cudaGetLastError());
  // Check the kernel ran correctly
  ASSERT_CUDA_SYNCHRONIZE();
  cuda_free(eigenvalues_dev);

  ScalarType *eigenvectors_dev = nullptr;
  cuda_error_code = cudaMalloc(&eigenvectors_dev,
                               n_eigenvectors * n_rows * sizeof(ScalarType));
  ASSERT_CUDA(cuda_error_code);
  n_blocks = 1 + (n_eigenvectors * n_rows - 1) / BLOCK_SIZE;
  internal::restrict_array<<<n_blocks, BLOCK_SIZE>>>(
      n_rows * n_rows, dense_system_matrix_dev, n_eigenvectors * n_rows,
      eigenvectors_dev);
  // Check that the kernel was launched correctly
  ASSERT_CUDA(cudaGetLastError());
  // Check the kernel ran correctly
  ASSERT_CUDA_SYNCHRONIZE();
  cuda_free(dense_system_matrix_dev);

  // Compute the map between the local and the global dof indices.
  std::vector<dealii::types::global_dof_index> dof_indices_map =
      this->compute_dof_index_map(patch_to_global_map, agglomerate_dof_handler);

  return std::make_tuple(smallest_eigenvalues_dev, eigenvectors_dev,
                         dof_indices_map);
}

template <int dim, typename VectorType>
SparseMatrixDevice<typename VectorType::value_type>
AMGe_device<dim, VectorType>::compute_restriction_sparse_matrix(
    ScalarType *eigenvectors_dev,
    std::vector<std::vector<dealii::types::global_dof_index>> const
        &dof_indices_maps)
{
  // The value in the sparse matrix are the same as the ones in the eigenvectors
  // so we just to compute the sparsity pattern.

  // TODO for now it doesn't work with MPI
  // dof_indices_maps contains all column indices, we just need to move them to
  // the GPU
  unsigned int const n_rows = dof_indices_maps.size();
  std::vector<int> row_ptr(n_rows + 1, 0);
  for (unsigned int i = 0; i < n_rows; ++i)
    row_ptr[i + 1] = row_ptr[i] + dof_indices_maps[i].size();
  int *row_ptr_dev;
  cudaError_t cuda_error = cudaMalloc(&row_ptr_dev, (n_rows + 1) * sizeof(int));
  ASSERT_CUDA(cuda_error);
  cuda_error = cudaMemcpy(row_ptr_dev, &row_ptr[0], (n_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice);
  ASSERT_CUDA(cuda_error);

  std::vector<int> column_index;
  column_index.reserve(row_ptr[n_rows]);
  for (unsigned int i = 0; i < n_rows; ++i)
    column_index.insert(column_index.end(), dof_indices_maps[i].begin(),
                        dof_indices_maps[i].end());
  int *column_index_dev;
  cuda_error = cudaMalloc(&column_index_dev, row_ptr[n_rows] * sizeof(int));
  ASSERT_CUDA(cuda_error);
  cuda_error =
      cudaMemcpy(column_index_dev, &column_index[0],
                 row_ptr[n_rows] * sizeof(int), cudaMemcpyHostToDevice);
  ASSERT_CUDA(cuda_error);

  return SparseMatrixDevice<ScalarType>(eigenvectors_dev, column_index_dev,
                                        row_ptr_dev, _cusparse_handle,
                                        row_ptr[n_rows], n_rows);
}
}

#endif
