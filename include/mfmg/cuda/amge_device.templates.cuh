/**************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                            *
 * All rights reserved.                                                   *
 *                                                                        *
 * This file is part of the mfmg library. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the  *
 * top-level directory                                                    *
 *                                                                        *
 * SPDX-License-Identifier: BSD-3-Clause                                  *
 *************************************************************************/

#ifndef AMGE_DEVICE_TEMPLATES_CUH
#define AMGE_DEVICE_TEMPLATES_CUH

#include <mfmg/common/amge.templates.hpp>
#include <mfmg/common/utils.hpp>
#include <mfmg/cuda/amge_device.cuh>
#include <mfmg/cuda/utils.cuh>

#include <deal.II/dofs/dof_accessor.h>

#include <omp.h>

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
  int n_rows = sparse_matrix_dev->m();

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
  int n_rows = sparse_matrix_dev->m();

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

  // NOTE: we add this code to make our library depend on gomp. Otherwise,
  // as we have not explicit references to OpenMP, we don't depend on it, which
  // results in linking errors with cuSolver as that also does not depend on it
  // for an unknown reason.
  std::ignore = omp_get_num_threads();

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
__global__ void extract_diag(ScalarType const *matrix, int n_rows, int n_cols,
                             ScalarType *diag_elements)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n_rows * n_cols)
    if ((i / n_rows) == (i % n_cols))
      diag_elements[i % n_cols] = matrix[i];
}

template <typename ScalarType>
__global__ void fill_identity_matrix(int n_rows, int n_cols, ScalarType *matrix)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n_rows * n_cols)
  {
    if ((i / n_rows) == (i % n_cols))
      matrix[i] = 1.;
    else
      matrix[i] = 0.;
  }
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
} // namespace internal

template <int dim, typename MeshEvaluator, typename VectorType>
AMGe_device<dim, MeshEvaluator, VectorType>::AMGe_device(
    MPI_Comm comm, dealii::DoFHandler<dim> const &dof_handler,
    CudaHandle const &cuda_handle)
    : AMGe<dim, VectorType>(comm, dof_handler), _cuda_handle(cuda_handle)
{
}

// Cannot be const because of the handles
template <int dim, typename MeshEvaluator, typename VectorType>
std::tuple<typename VectorType::value_type *, typename VectorType::value_type *,
           typename VectorType::value_type *,
           std::vector<dealii::types::global_dof_index>>
AMGe_device<dim, MeshEvaluator, VectorType>::compute_local_eigenvectors(
    unsigned int n_eigenvectors,
    dealii::Triangulation<dim> const &agglomerate_triangulation,
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator> const
        &patch_to_global_map,
    MeshEvaluator const &evaluator)
{
  dealii::DoFHandler<dim> agglomerate_dof_handler(agglomerate_triangulation);
  dealii::AffineConstraints<double> agglomerate_constraints;

  // Call user function to build the system matrix
  using value_type = typename VectorType::value_type;
  auto agglomerate_system_matrix_dev =
      std::make_shared<SparseMatrixDevice<value_type>>();
  evaluator.evaluate_agglomerate(agglomerate_dof_handler,
                                 agglomerate_constraints,
                                 *agglomerate_system_matrix_dev);

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
  int const n_rows = agglomerate_system_matrix_dev->m();
  int const n_cols = agglomerate_system_matrix_dev->n();
  ASSERT(n_cols == n_rows,
         "The system matrix on the agglomerate is not square.");

  // Convert the system matrix to dense
  ScalarType *dense_system_matrix_dev = nullptr;
  internal::convert_csr_to_dense(_cuda_handle.cusparse_handle, descr,
                                 agglomerate_system_matrix_dev,
                                 dense_system_matrix_dev);
  // Free the memory of the system sparse matrix
  agglomerate_system_matrix_dev.reset();

  // Get the diagonal elements
  ScalarType *diag_elements_dev = nullptr;
  cuda_malloc(diag_elements_dev, n_rows);
  int n_blocks = 1 + (n_rows * n_cols - 1) / block_size;
  internal::extract_diag<<<n_blocks, block_size>>>(
      dense_system_matrix_dev, n_rows, n_cols, diag_elements_dev);

  // Create the dense mass matrix
  ScalarType *dense_mass_matrix_dev = nullptr;
  cuda_malloc(dense_mass_matrix_dev, n_rows * n_cols);
  n_blocks = 1 + (n_rows * n_cols - 1) / block_size;
  internal::fill_identity_matrix<<<n_blocks, block_size>>>(
      n_rows, n_cols, dense_mass_matrix_dev);

  // Compute the eigenvalues and the eigenvectors. The values in
  // dense_system_matrix_dev are overwritten and replaced by the eigenvectors
  ScalarType *eigenvalues_dev = nullptr;
  cudaError_t cuda_error_code;
  cuda_error_code = cudaMalloc(&eigenvalues_dev, n_rows * sizeof(ScalarType));
  internal::compute_local_eigenvectors(_cuda_handle.cusolver_dn_handle, n_rows,
                                       dense_system_matrix_dev,
                                       dense_mass_matrix_dev, eigenvalues_dev);
  cuda_free(dense_mass_matrix_dev);
  // We now have too many eigenvectors. So we only keep the ones associated to
  // the smallest ones.
  ScalarType *smallest_eigenvalues_dev = nullptr;
  cuda_error_code = cudaMalloc(&smallest_eigenvalues_dev,
                               n_eigenvectors * sizeof(ScalarType));

  ASSERT_CUDA(cuda_error_code);
  n_blocks = 1 + (n_eigenvectors - 1) / block_size;
  internal::restrict_array<<<n_blocks, block_size>>>(
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
  n_blocks = 1 + (n_eigenvectors * n_rows - 1) / block_size;
  internal::restrict_array<<<n_blocks, block_size>>>(
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
                         diag_elements_dev, dof_indices_map);
}

template <int dim, typename MeshEvaluator, typename VectorType>
SparseMatrixDevice<typename VectorType::value_type>
AMGe_device<dim, MeshEvaluator, VectorType>::compute_restriction_sparse_matrix(
    std::vector<dealii::Vector<typename VectorType::value_type>> const
        &eigenvectors,
    std::vector<std::vector<typename VectorType::value_type>> const
        &diag_elements,
    dealii::LinearAlgebra::distributed::Vector<
        typename VectorType::value_type> const &locally_relevant_global_diag,
    std::vector<std::vector<dealii::types::global_dof_index>> const
        &dof_indices_maps,
    std::vector<unsigned int> const &n_local_eigenvectors,
    cusparseHandle_t cusparse_handle)
{
  dealii::TrilinosWrappers::SparseMatrix restriction_sparse_matrix;
  AMGe<dim, VectorType>::compute_restriction_sparse_matrix(
      eigenvectors, diag_elements, dof_indices_maps, n_local_eigenvectors,
      locally_relevant_global_diag, restriction_sparse_matrix);
  check_restriction_matrix(this->_comm, eigenvectors, dof_indices_maps,
                           locally_relevant_global_diag, diag_elements,
                           n_local_eigenvectors);

  SparseMatrixDevice<ScalarType> restriction_sparse_matrix_dev(
      convert_matrix(restriction_sparse_matrix));

  restriction_sparse_matrix_dev.cusparse_handle = cusparse_handle;
  cusparseStatus_t cusparse_error_code;
  cusparse_error_code =
      cusparseCreateMatDescr(&restriction_sparse_matrix_dev.descr);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code = cusparseSetMatType(restriction_sparse_matrix_dev.descr,
                                           CUSPARSE_MATRIX_TYPE_GENERAL);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code = cusparseSetMatIndexBase(
      restriction_sparse_matrix_dev.descr, CUSPARSE_INDEX_BASE_ZERO);
  ASSERT_CUSPARSE(cusparse_error_code);

  return std::move(restriction_sparse_matrix_dev);
}

template <int dim, typename MeshEvaluator, typename VectorType>
mfmg::SparseMatrixDevice<typename VectorType::value_type>
AMGe_device<dim, MeshEvaluator, VectorType>::setup_restrictor(
    boost::property_tree::ptree const &agglomerate_dim,
    unsigned int const n_eigenvectors, double const tolerance,
    MeshEvaluator const &evaluator)
{
  // Flag the cells to build agglomerates.
  unsigned int const n_agglomerates = this->build_agglomerates(agglomerate_dim);

  std::vector<unsigned int> agglomerate_ids(n_agglomerates);
  std::iota(agglomerate_ids.begin(), agglomerate_ids.end(), 1);
  std::vector<dealii::Vector<double>> eigenvectors;
  std::vector<std::vector<ScalarType>> diag_elements;
  std::vector<std::vector<dealii::types::global_dof_index>> dof_indices_maps;
  std::vector<unsigned int> n_local_eigenvectors;
  for (auto const &agg_id : agglomerate_ids)
  {
    dealii::Triangulation<dim> agglomerate_triangulation;
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator>
        agglomerate_to_global_tria_map;

    this->build_agglomerate_triangulation(agg_id, agglomerate_triangulation,
                                          agglomerate_to_global_tria_map);

    // TODO this should be batched because unless the agglomerate are very
    // large, the matrices won't fill up the GPU We ignore the eigenvalues.
    ScalarType *eigenvectors_dev = nullptr;
    ScalarType *diag_elements_dev = nullptr;
    std::vector<dealii::types::global_dof_index> local_dof_indices_map;
    std::tie(std::ignore, eigenvectors_dev, diag_elements_dev,
             local_dof_indices_map) =
        compute_local_eigenvectors(n_eigenvectors, agglomerate_triangulation,
                                   agglomerate_to_global_tria_map, evaluator);

    // Move the data to the host and reformat it.
    unsigned int const n_local_dof_indices = local_dof_indices_map.size();
    std::vector<ScalarType> eigenvectors_host(n_eigenvectors *
                                              n_local_dof_indices);
    cuda_mem_copy_to_host(eigenvectors_dev, eigenvectors_host);
    for (unsigned int i = 0; i < n_eigenvectors; ++i)
    {
      unsigned int const begin_offset = i * n_local_dof_indices;
      unsigned int const end_offset = (i + 1) * n_local_dof_indices;
      eigenvectors.emplace_back(eigenvectors_host.begin() + begin_offset,
                                eigenvectors_host.begin() + end_offset);
    }
    cuda_free(eigenvectors_dev);

    std::vector<ScalarType> diag_elements_host(n_local_dof_indices);
    cuda_mem_copy_to_host(diag_elements_dev, diag_elements_host);
    diag_elements.push_back(diag_elements_host);
    cuda_free(diag_elements_dev);

    dof_indices_maps.push_back(local_dof_indices_map);

    n_local_eigenvectors.push_back(n_eigenvectors);
  }

  // Get the locally relevant global diagonal
  auto locally_relevant_global_diag = evaluator.get_locally_relevant_diag();

  return compute_restriction_sparse_matrix(
      eigenvectors, diag_elements, locally_relevant_global_diag,
      dof_indices_maps, n_local_eigenvectors,
      evaluator.get_cuda_handle().cusparse_handle);
}
} // namespace mfmg

#endif
