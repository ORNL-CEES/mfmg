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
#include <mfmg/common/lanczos.templates.hpp>
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

struct LocallyRelevantDiagonal
{
  template <
      typename MeshEvaluator,
      std::enable_if_t<mfmg::is_matrix_free<MeshEvaluator>::value, int> = 0>
  static dealii::LinearAlgebra::distributed::Vector<double,
                                                    dealii::MemorySpace::Host>
  get(MeshEvaluator const &evaluator)
  {
    return mfmg::copy_from_dev(evaluator.get_diagonal());
  }

  template <
      typename MeshEvaluator,
      std::enable_if_t<!mfmg::is_matrix_free<MeshEvaluator>::value, int> = 0>
  static dealii::LinearAlgebra::distributed::Vector<double,
                                                    dealii::MemorySpace::Host>
  get(MeshEvaluator const &evaluator)
  {
    return evaluator.get_locally_relevant_diag();
  }
};
} // namespace internal

/**
 * A class representing the operator on a given agglomerate
 * wrapping a MeshEvaluator object to restrict the interface
 * to the operations required on an agglomerate.
 */
template <typename MeshEvaluator>
struct CudaMatrixFreeAgglomerateOperator
{
  using size_type = typename MeshEvaluator::size_type;

  /**
   * This constructor expects @p mesh_evaluator to be an object that performs
   * the actual operator evaluation. @p dof_handler is used to initialize the
   * appropriate data structures in the MeshEvaluator and is expected to be
   * initialized itself with respect to a given dealii::FiniteElement object
   * in that call.
   */
  template <typename DoFHandler>
  CudaMatrixFreeAgglomerateOperator(
      MeshEvaluator const &mesh_evaluator, DoFHandler &dof_handler,
      dealii::AffineConstraints<double> &constraints)
      : _mesh_evaluator(mesh_evaluator), _dof_handler(dof_handler),
        _constraints(constraints)
  {
    _mesh_evaluator.matrix_free_initialize_agglomerate(dof_handler);
  }

  /**
   * Perform the operator evaluation on the agglomerate.
   */
  void vmult(dealii::LinearAlgebra::distributed::Vector<
                 double, dealii::MemorySpace::CUDA> &dst,
             dealii::LinearAlgebra::distributed::Vector<
                 double, dealii::MemorySpace::CUDA> const &src) const
  {
    _mesh_evaluator.matrix_free_evaluate_agglomerate(src, dst);
  }

  /**
   * Return the diagonal entries the matrix corresponding to the operator would
   * have. This data is necessary for certain smoothers to work.
   * @p _constraints is used to restrict the diagonal to the correct
   * (constrained) finite element subspace.
   */
  std::vector<double> get_diag_elements() const
  {
    return _mesh_evaluator.matrix_free_get_agglomerate_diagonal(_constraints);
  }

  /**
   * Return the dimension of the range of the agglomerate operator.
   */
  size_type m() const { return _dof_handler.n_dofs(); }

  /**
   * Return the dimension of the domain of the agglomerate operator.
   */
  size_type n() const { return _dof_handler.n_dofs(); }

private:
  /**
   * The actual operator wrapped.
   */
  MeshEvaluator const &_mesh_evaluator;

  /**
   * The dimension for the underlying mesh.
   */
  static int constexpr dim = MeshEvaluator::_dim;

  /**
   * The DoFHandler containing information about the degrees of freedom on the
   * agglomerate.
   */
  dealii::DoFHandler<dim> const &_dof_handler;

  /**
   * The constraints needed for restricting the vector returned by
   * get_diag_elements() to the correct subspace.
   */
  dealii::AffineConstraints<double> &_constraints;
};

template <int dim, typename MeshEvaluator, typename VectorType>
AMGe_device<dim, MeshEvaluator, VectorType>::AMGe_device(
    MPI_Comm comm, dealii::DoFHandler<dim> const &dof_handler,
    CudaHandle const &cuda_handle,
    boost::property_tree::ptree const &eigensolver_params)
    : AMGe<dim, VectorType>(comm, dof_handler), _cuda_handle(cuda_handle),
      _eigensolver_params(eigensolver_params)
{
}

template <int dim, typename MeshEvaluator, typename VectorType>
template <typename TriangulationType>
std::tuple<typename VectorType::value_type *, typename VectorType::value_type *,
           typename VectorType::value_type *,
           std::vector<dealii::types::global_dof_index>>
AMGe_device<dim, MeshEvaluator, VectorType>::compute_local_eigenvectors(
    unsigned int n_eigenvectors, double const tolerance,
    TriangulationType const &agglomerate_triangulation,
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator> const
        &patch_to_global_map,
    MeshEvaluator const &evaluator,
    typename std::enable_if_t<is_matrix_free<MeshEvaluator>::value &&
                                  std::is_class<TriangulationType>::value,
                              int>)
{
  dealii::DoFHandler<dim> agglomerate_dof_handler(agglomerate_triangulation);
  dealii::AffineConstraints<double> agglomerate_constraints;

  using AgglomerateOperator = CudaMatrixFreeAgglomerateOperator<MeshEvaluator>;
  AgglomerateOperator agglomerate_operator(evaluator, agglomerate_dof_handler,
                                           agglomerate_constraints);

  auto const diag_elements = agglomerate_operator.get_diag_elements();

  // Compute the eigenvalues and the eigenvectors
  unsigned int const n_dofs_agglomerate = agglomerate_dof_handler.n_dofs();
  std::vector<dealii::LinearAlgebra::distributed::Vector<
      double, dealii::MemorySpace::CUDA>>
  eigenvectors(n_eigenvectors,
               dealii::LinearAlgebra::distributed::Vector<
                   double, dealii::MemorySpace::CUDA>(n_dofs_agglomerate));
  dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::CUDA>
      initial_vector(n_dofs_agglomerate);
  evaluator.set_initial_guess(agglomerate_constraints, initial_vector);

  boost::property_tree::ptree lanczos_params;
  lanczos_params.put("num_eigenpairs", n_eigenvectors);
  // We are having trouble with Lanczos when tolerance is too tight.
  // Typically, it results in spurious eigenvalues (so far, only noticed 0).
  // This seems to be the result of producing too many Lanczos vectors. For
  // hierarchy_3d tests, any tolerance below 1e-5 (e.g., 1e-6) produces this
  // problem. Thus, we try to work around it here. It is still unclear how
  // robust this is.
  lanczos_params.put("tolerance", std::max(tolerance, 1e-4));
  lanczos_params.put("max_iterations",
                     _eigensolver_params.get("max_iterations", 200));
  lanczos_params.put("percent_overshoot",
                     _eigensolver_params.get("percent_overshoot", 5));
  bool is_deflated = _eigensolver_params.get("is_deflated", false);
  if (is_deflated)
  {
    lanczos_params.put("is_deflated", true);
    lanczos_params.put("num_cycles",
                       _eigensolver_params.get<int>("num_cycles"));
    lanczos_params.put(
        "num_eigenpairs_per_cycle",
        _eigensolver_params.get<int>("num_eigenpairs_per_cycle"));
  }

  Lanczos<AgglomerateOperator, dealii::LinearAlgebra::distributed::Vector<
                                   double, dealii::MemorySpace::CUDA>>
      solver(agglomerate_operator);

  std::vector<double> eigenvalues;
  std::tie(eigenvalues, eigenvectors) =
      solver.solve(lanczos_params, initial_vector);
  ASSERT(n_eigenvectors == eigenvectors.size(),
         "Wrong number of computed eigenpairs");

  // Compute the map between the local and the global dof indices.
  std::vector<dealii::types::global_dof_index> dof_indices_map =
      this->compute_dof_index_map(patch_to_global_map, agglomerate_dof_handler);

  // Move the eigenvalues to the device
  double *eigenvalues_dev = nullptr;
  cudaError_t cuda_error_code =
      cudaMalloc(&eigenvalues_dev, n_eigenvectors * sizeof(double));
  ASSERT_CUDA(cuda_error_code);
  cuda_error_code =
      cudaMemcpy(eigenvalues_dev, eigenvalues.data(),
                 n_eigenvectors * sizeof(double), cudaMemcpyHostToDevice);
  ASSERT_CUDA(cuda_error_code);

  // Move the eigenvectors to the device
  double *eigenvectors_dev = nullptr;
  cuda_error_code = cudaMalloc(
      &eigenvectors_dev, n_eigenvectors * n_dofs_agglomerate * sizeof(double));
  ASSERT_CUDA(cuda_error_code);
  for (unsigned int i = 0; i < n_eigenvectors; ++i)
  {
    cuda_error_code = cudaMemcpy(
        eigenvectors_dev + i * n_dofs_agglomerate, eigenvectors[i].get_values(),
        n_dofs_agglomerate * sizeof(double), cudaMemcpyDeviceToDevice);
    ASSERT_CUDA(cuda_error_code);
  }
  ScalarType *diag_elements_dev = nullptr;
  cuda_malloc(diag_elements_dev, n_dofs_agglomerate);
  cuda_mem_copy_to_dev(diag_elements, diag_elements_dev);

  return std::make_tuple(eigenvalues_dev, eigenvectors_dev, diag_elements_dev,
                         dof_indices_map);
}

// Cannot be const because of the handler
template <int dim, typename MeshEvaluator, typename VectorType>
template <typename TriangulationType>
std::tuple<typename VectorType::value_type *, typename VectorType::value_type *,
           typename VectorType::value_type *,
           std::vector<dealii::types::global_dof_index>>
AMGe_device<dim, MeshEvaluator, VectorType>::compute_local_eigenvectors(
    unsigned int n_eigenvectors, double const /*tolerance*/,
    TriangulationType const &agglomerate_triangulation,
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator> const
        &patch_to_global_map,
    MeshEvaluator const &evaluator,
    typename std::enable_if_t<!is_matrix_free<MeshEvaluator>::value &&
                                  std::is_class<TriangulationType>::value,
                              int>)
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

  // When checking the restriction matrix, we check that the sum of the local
  // diagonals is the global diagonals. This is not true for matrix-free because
  // the constraints values are set arbitrarily.
  if (is_matrix_free<MeshEvaluator>::value == false)
  {
    check_restriction_matrix(this->_comm, eigenvectors, dof_indices_maps,
                             locally_relevant_global_diag, diag_elements,
                             n_local_eigenvectors);
  }

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
        compute_local_eigenvectors(n_eigenvectors, tolerance,
                                   agglomerate_triangulation,
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
  auto locally_relevant_global_diag =
      internal::LocallyRelevantDiagonal::get(evaluator);
  locally_relevant_global_diag.update_ghost_values();

  return compute_restriction_sparse_matrix(
      eigenvectors, diag_elements, locally_relevant_global_diag,
      dof_indices_maps, n_local_eigenvectors,
      evaluator.get_cuda_handle().cusparse_handle);
}
} // namespace mfmg

#endif
