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

#include <deal.II/base/mpi.h>
#include <mfmg/dealii_operator_device.cuh>
#include <mfmg/dealii_operator_device_helpers.cuh>
#include <mfmg/utils.cuh>
// This is only tmp this files has moved in deal.II 9.0
#include <deal.II/base/mpi.templates.h>

#include <EpetraExt_Transpose_RowMatrix.h>

#include <algorithm>

namespace mfmg
{
namespace internal
{
template <typename VectorType>
struct MatrixOperator
{
  static void
  apply(std::shared_ptr<SparseMatrixDevice<typename VectorType::value_type>>
            matrix,
        VectorType const &x, VectorType &y);
};

template <typename VectorType>
void MatrixOperator<VectorType>::apply(
    std::shared_ptr<SparseMatrixDevice<typename VectorType::value_type>> matrix,
    VectorType const &x, VectorType &y)
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <>
void MatrixOperator<VectorDevice<double>>::apply(
    std::shared_ptr<SparseMatrixDevice<double>> matrix,
    VectorDevice<double> const &x, VectorDevice<double> &y)
{
  matrix->vmult(y, x);
}

template <>
void MatrixOperator<dealii::LinearAlgebra::distributed::Vector<double>>::apply(
    std::shared_ptr<SparseMatrixDevice<double>> matrix,
    dealii::LinearAlgebra::distributed::Vector<double> const &x,
    dealii::LinearAlgebra::distributed::Vector<double> &y)
{
  // Move the data to the device
  VectorDevice<double> x_dev(x);
  VectorDevice<double> y_dev(y);

  matrix->vmult(y_dev, x_dev);

  // Move the data to the host
  std::vector<double> y_host(y.local_size());
  cuda_mem_copy_to_host(y_dev.val_dev, y_host);
  std::copy(y_host.begin(), y_host.end(), y.begin());
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

template <typename VectorType>
struct DirectOperator
{
  static void
  apply(CudaHandle const &cuda_handle,
        SparseMatrixDevice<typename VectorType::value_type> const &matrix,
        std::string const &solver, VectorType const &b, VectorType &x);

#if MFMG_WITH_AMGX
  static void
  amgx_solve(std::unordered_map<int, int> const &row_map,
             AMGX_vector_handle const &amgx_rhs_handle,
             AMGX_vector_handle const &amgx_solution_handle,
             AMGX_solver_handle const &amgx_solver_handle,
             SparseMatrixDevice<typename VectorType::value_type> const &matrix,
             VectorType &b, VectorType &x);
#endif
};

template <typename VectorType>
void DirectOperator<VectorType>::apply(
    CudaHandle const &cuda_handle,
    SparseMatrixDevice<typename VectorType::value_type> const &matrix,
    std::string const &solver, VectorType const &b, VectorType &x)
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <>
void DirectOperator<VectorDevice<double>>::apply(
    CudaHandle const &cuda_handle, SparseMatrixDevice<double> const &matrix,
    std::string const &solver, VectorDevice<double> const &b,
    VectorDevice<double> &x)
{
  if (solver == "cholesky")
    cholesky_factorization(cuda_handle.cusolver_sp_handle, matrix,
                           b.get_values(), x.get_values());
  else if (solver == "lu_dense")
    lu_factorization(cuda_handle.cusolver_dn_handle, matrix, b.get_values(),
                     x.get_values());
  else if (solver == "lu_sparse_host")
    lu_factorization(cuda_handle.cusolver_sp_handle, matrix, b.get_values(),
                     x.get_values());
  else
    ASSERT_THROW(false, "The provided solver name " + solver + " is invalid.");
}

template <>
void DirectOperator<dealii::LinearAlgebra::distributed::Vector<double>>::apply(
    CudaHandle const &cuda_handle, SparseMatrixDevice<double> const &matrix,
    std::string const &solver,
    dealii::LinearAlgebra::distributed::Vector<double> const &b,
    dealii::LinearAlgebra::distributed::Vector<double> &x)
{
  // Copy to the device
  VectorDevice<double> x_dev(x);
  VectorDevice<double> b_dev(b);

  DirectOperator<VectorDevice<double>>::apply(cuda_handle, matrix, solver,
                                              b_dev, x_dev);

  // Move the data to the host
  std::vector<double> x_host(x.local_size());
  cuda_mem_copy_to_host(x_dev.val_dev, x_host);
  std::copy(x_host.begin(), x_host.end(), x.begin());
}

#if MFMG_WITH_AMGX
template <typename VectorType>
void DirectOperator<VectorType>::amgx_solve(
    std::unordered_map<int, int> const &row_map,
    AMGX_vector_handle const &amgx_rhs_handle,
    AMGX_vector_handle const &amgx_solution_handle,
    AMGX_solver_handle const &amgx_solver_handle,
    SparseMatrixDevice<typename VectorType::value_type> const &matrix,
    VectorType &b, VectorType &x)
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <>
void DirectOperator<VectorDevice<double>>::amgx_solve(
    std::unordered_map<int, int> const &row_map,
    AMGX_vector_handle const &amgx_rhs_handle,
    AMGX_vector_handle const &amgx_solution_handle,
    AMGX_solver_handle const &amgx_solver_handle,
    SparseMatrixDevice<double> const &matrix, VectorDevice<double> &b,
    VectorDevice<double> &x)
{
  // Copy data into a vector object
  unsigned int const n_local_rows = matrix.n_local_rows();
  std::vector<double> val_host(n_local_rows);
  mfmg::cuda_mem_copy_to_host(b.val_dev, val_host);
  std::vector<double> tmp(val_host);
  for (auto const &pos : row_map)
    val_host[pos.second] = tmp[pos.first];
  mfmg::cuda_mem_copy_to_dev(val_host, b.val_dev);

  int const block_dim_x = 1;

  AMGX_vector_upload(amgx_rhs_handle, n_local_rows, block_dim_x, b.val_dev);
  AMGX_vector_upload(amgx_solution_handle, n_local_rows, block_dim_x,
                     x.val_dev);

  // Solve the problem
  AMGX_solver_solve(amgx_solver_handle, amgx_rhs_handle, amgx_solution_handle);

  // Get the result back
  AMGX_vector_download(amgx_solution_handle, x.val_dev);

  // Move the result back to the host to reorder it
  std::vector<double> solution_host(n_local_rows);
  mfmg::cuda_mem_copy_to_host(x.val_dev, solution_host);
  tmp = solution_host;
  for (auto const &pos : row_map)
    solution_host[pos.first] = tmp[pos.second];

  // Move the solution back on the device
  mfmg::cuda_mem_copy_to_dev(solution_host, x.val_dev);
}

template <>
void DirectOperator<dealii::LinearAlgebra::distributed::Vector<double>>::
    amgx_solve(std::unordered_map<int, int> const &row_map,
               AMGX_vector_handle const &amgx_rhs_handle,
               AMGX_vector_handle const &amgx_solution_handle,
               AMGX_solver_handle const &amgx_solver_handle,
               SparseMatrixDevice<double> const &matrix,
               dealii::LinearAlgebra::distributed::Vector<double> &b,
               dealii::LinearAlgebra::distributed::Vector<double> &x)
{
  auto partitioner = b.get_partitioner();
  VectorDevice<double> x_dev(partitioner);
  VectorDevice<double> b_dev(partitioner);

  unsigned int const n_local_rows = matrix.n_local_rows();
  std::vector<double> val_host(n_local_rows);
  std::copy(b.begin(), b.end(), val_host.begin());
  std::vector<double> tmp(val_host);
  for (auto const &pos : row_map)
    val_host[pos.second] = tmp[pos.first];
  mfmg::cuda_mem_copy_to_dev(val_host, b_dev.val_dev);

  int const block_dim_x = 1;

  AMGX_vector_upload(amgx_rhs_handle, n_local_rows, block_dim_x, b_dev.val_dev);
  AMGX_vector_upload(amgx_solution_handle, n_local_rows, block_dim_x,
                     x_dev.val_dev);

  // Solve the problem
  AMGX_solver_solve(amgx_solver_handle, amgx_rhs_handle, amgx_solution_handle);

  // Get the result back
  AMGX_vector_download(amgx_solution_handle, x_dev.val_dev);

  // Move the result back to the host to reorder it
  std::vector<double> solution_host(n_local_rows);
  mfmg::cuda_mem_copy_to_host(x_dev.val_dev, solution_host);
  tmp = solution_host;
  for (auto const &pos : row_map)
    x.local_element(pos.first) = tmp[pos.second];
}
#endif
} // namespace internal

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
  internal::MatrixOperator<VectorType>::apply(_matrix, x, y);
}

template <typename VectorType>
std::shared_ptr<MatrixOperator<VectorType>>
SparseMatrixDeviceOperator<VectorType>::transpose() const
{
  // TODO: to do this on the GPU
  // Copy the data to the cpu and then, use trilinos to compute the
  // transpose. This is not the most efficient way to do this but it is the
  // easiest.
  auto sparse_matrix = convert_to_trilinos_matrix(*_matrix);

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
  internal::SmootherOperator<VectorType>::apply(_matrix, _smoother, b, x);
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

  int n_blocks = 1 + (local_nnz - 1) / block_size;
  internal::extract_inv_diag<<<n_blocks, block_size>>>(
      _matrix.val_dev, _matrix.column_index_dev, row_index_coo_dev, local_nnz,
      val_dev);

  iota<<<n_blocks, block_size>>>(size, column_index_dev);

  n_blocks = 1 + size / block_size;
  iota<<<n_blocks, block_size>>>(size + 1, row_ptr_dev);

  _smoother.reinit(_matrix.get_mpi_communicator(), val_dev, column_index_dev,
                   row_ptr_dev, size, _matrix.locally_owned_range_indices(),
                   _matrix.locally_owned_range_indices(),
                   _matrix.cusparse_handle);

  cuda_free(row_index_coo_dev);
}

//-------------------------------------------------------------------------//

template <typename VectorType>
DirectDeviceOperator<VectorType>::DirectDeviceOperator(
    CudaHandle const &cuda_handle,
    SparseMatrixDevice<typename VectorType::value_type> const &matrix,
    std::shared_ptr<boost::property_tree::ptree> params)
    : _cuda_handle(cuda_handle), _matrix(matrix), _amgx_config_handle(nullptr),
      _amgx_res_handle(nullptr), _amgx_matrix_handle(nullptr),
      _amgx_rhs_handle(nullptr), _amgx_solution_handle(nullptr),
      _amgx_solver_handle(nullptr)
{
  // Transform to lower case
  _solver = params->get("solver.type", "lu_dense");
  std::transform(_solver.begin(), _solver.end(), _solver.begin(), tolower);

#if MFMG_WITH_AMGX
  // If the solver is amgx, we need the name of the input file
  if (_solver == "amgx")
  {
    _amgx_config_file = params->get<std::string>("solver.config_file");

    // We need to cast away the const because we need to change the format of
    // _matrix to the format supported by amgx
    auto &amgx_matrix =
        const_cast<SparseMatrixDevice<typename VectorType::value_type> &>(
            _matrix);

    AMGX_initialize();
    AMGX_initialize_plugins();

    AMGX_config_create_from_file(&_amgx_config_handle,
                                 _amgx_config_file.data());

    int current_device_id = 0;
    cudaError_t cuda_error_code = cudaGetDevice(&current_device_id);
    ASSERT_CUDA(cuda_error_code);
    _device_id[0] = current_device_id;

    MPI_Comm comm = amgx_matrix.get_mpi_communicator();
    AMGX_resources_create(&_amgx_res_handle, _amgx_config_handle, &comm, 1,
                          _device_id);

    // Mode: d(evice) D(ouble precision for the matrix) D(ouble precision for
    // the vector) I(nteger index)
    AMGX_Mode amgx_mode = AMGX_mode_dDDI;

    // Create the matrix object
    AMGX_matrix_create(&_amgx_matrix_handle, _amgx_res_handle, amgx_mode);

    // Create the rhs object
    AMGX_vector_create(&_amgx_rhs_handle, _amgx_res_handle, amgx_mode);

    // Create the solution object
    AMGX_vector_create(&_amgx_solution_handle, _amgx_res_handle, amgx_mode);

    // Create the solver object
    AMGX_solver_create(&_amgx_solver_handle, _amgx_res_handle, amgx_mode,
                       _amgx_config_handle);

    // Set the communication map
    // First we communicate all the data needed to build the communication
    // map. This means for each processor the global IDs of the rows owned,
    // row_ptr, and column_indices. We will need a buffer of size
    // 2*n_local_rows + local_nnz

    // Get the global IDs of the rows
    std::vector<unsigned int> global_rows;
    auto range_indexset = amgx_matrix.locally_owned_range_indices();
    range_indexset.fill_index_vector(global_rows);

    // Move row_ptr_dev to the host
    int const n_local_rows = amgx_matrix.n_local_rows();
    std::vector<int> row_ptr_host(n_local_rows);
    mfmg::cuda_mem_copy_to_host(amgx_matrix.row_ptr_dev, row_ptr_host);

    // Move column_index_dev to the host
    int local_nnz = amgx_matrix.local_nnz();
    std::vector<int> column_index_host(local_nnz);
    mfmg::cuda_mem_copy_to_host(amgx_matrix.column_index_dev,
                                column_index_host);

    // Create and fill the buffer
    std::vector<int> sparsity_pattern_buffer(2);
    sparsity_pattern_buffer[0] = n_local_rows;
    sparsity_pattern_buffer[1] = local_nnz;
    sparsity_pattern_buffer.insert(sparsity_pattern_buffer.end(),
                                   range_indexset.begin(),
                                   range_indexset.end());
    sparsity_pattern_buffer.insert(sparsity_pattern_buffer.end(),
                                   column_index_host.begin(),
                                   column_index_host.end());

    auto global_sparsity_pattern = dealii::Utilities::MPI::all_gather(
        MPI_COMM_WORLD, sparsity_pattern_buffer);

    // Now we have all the data. We can compute what is required by AMGX
    // Search in global_sparsity_pattern the columns corresponding the locally
    // owned rows. Because we don't want to search for the index of every row
    // among every single entry, we first use an unordered_set to find which
    // processors contain interesting data.
    std::vector<int> neighbors;
    std::vector<std::vector<int>> row_halo;
    std::unordered_set<int> rows_sent;
    int const comm_size = dealii::Utilities::MPI::n_mpi_processes(comm);
    int const rank = dealii::Utilities::MPI::this_mpi_process(comm);
    for (int i = 0; i < comm_size; ++i)
    {
      if (i != rank)
      {
        unsigned int const col_id_offset = 2 + global_sparsity_pattern[i][0];
        std::unordered_set<int> global_columns(
            global_sparsity_pattern[i].begin() + col_id_offset,
            global_sparsity_pattern[i].end());
        std::vector<int> halo;
        for (unsigned int j = 0; j < n_local_rows; ++j)
        {
          if (global_columns.count(global_rows[j]) == 1)
          {
            halo.emplace_back(j);
            rows_sent.insert(j);
          }
        }

        if (halo.size() > 0)
        {
          neighbors.emplace_back(i);
          row_halo.emplace_back(halo);
        }
      }
    }

    // With these data we can start populating the data structures required by
    // AMGX

    // Halo depth is zero in serial and one otherwise
    int const halo_depth = (comm_size == 1) ? 0 : 1;
    // Number of MPI ranks wich will exchange data via halo exchanges
    int const n_neighbors = neighbors.size();
    // The array of size n_neighbors which lists the MPI ranks which will
    // exchange data via halo exchange is the neighbors vector computed
    // before.

    // An array of size n_neighbors. The value in entry i is the number of
    // local rows in this rank's matrix partition which will be sent to the
    // MPI rank neighbors[i].
    std::vector<int> send_sizes(n_neighbors);
    for (int i = 0; i < n_neighbors; ++i)
      send_sizes[i] = row_halo[i].size();

    // We have filled the data structures of what we need to send. Now, we
    // need to fill the data structures for what we will receive.
    std::vector<std::vector<int>> recv_halo;
    for (int i = 0; i < comm_size; ++i)
    {
      if (i != rank)
      {
        // We recreate the IndexSet because searching if a value is inside an
        // IndexSet is very fast
        dealii::IndexSet indexset(amgx_matrix.m());
        indexset.add_indices(global_sparsity_pattern[i].begin() + 2,
                             global_sparsity_pattern[i].begin() + 2 +
                                 global_sparsity_pattern[i][0]);
        indexset.compress();
        std::set<int> halo;
        for (unsigned int j = 0; j < local_nnz; ++j)
        {
          if (indexset.is_element(column_index_host[j]))
          {
            halo.insert(column_index_host[j]);
          }
        }

        if (halo.size() > 0)
        {
          recv_halo.emplace_back(halo.begin(), halo.end());
        }
      }
    }

    // An array of size n_neighbors. The value in entry i is the number of
    // non-local rows in this rank's matrix partition which will be received
    // from the MPI rank neighbors[i]
    std::vector<int> recv_sizes(n_neighbors);
    for (int i = 0; i < n_neighbors; ++i)
      recv_sizes[i] = recv_halo[i].size();

    // Change the format of the matrix
    std::unordered_map<int, int> halo_map;
    std::tie(halo_map, _row_map) = mfmg::csr_to_amgx(rows_sent, amgx_matrix);

    // Because the format of the matrix has been changed, we need to update
    // the communication structure
    for (unsigned int i = 0; i < n_neighbors; ++i)
    {
      for (unsigned int j = 0; j < send_sizes[i]; ++j)
        row_halo[i][j] = _row_map[row_halo[i][j]];

      for (unsigned int j = 0; j < recv_sizes[i]; ++j)
        recv_halo[i][j] = halo_map[recv_halo[i][j]];
    }

    // An array of size n_neighbors of arrays, where entry i is another array
    // of size send_sizes[i]. Array i is a map specifying the local row
    // indices from this matrix partition which will be sent to the MPI rank
    // neighbors[i].
    std::vector<int const *> send_maps(n_neighbors);
    for (int i = 0; i < n_neighbors; ++i)
      send_maps[i] = row_halo[i].data();

    // An array of size n_neighbors of arrays, where entry i is another array
    // of size recv_sizes[i]. Array i is a map specifying the local halo
    // indices from this matrix partition which will be received from the MPI
    // rank neighbors[i].
    std::vector<int const *> recv_maps(n_neighbors);
    for (int i = 0; i < n_neighbors; ++i)
      recv_maps[i] = recv_halo[i].data();

    AMGX_matrix_comm_from_maps_one_ring(_amgx_matrix_handle, halo_depth,
                                        n_neighbors, neighbors.data(),
                                        send_sizes.data(), send_maps.data(),
                                        recv_sizes.data(), recv_maps.data());

    // Create the communication maps and partition info on a vector by copying
    // them from a matrix
    AMGX_vector_bind(_amgx_rhs_handle, _amgx_matrix_handle);
    AMGX_vector_bind(_amgx_solution_handle, _amgx_matrix_handle);

    // Copy the local matrix into a matrix object. When using one processor,
    // we can use CSR directly. Otherwise we need to reorder the matrix.
    int const block_dim_x = 1;
    int const block_dim_y = 1;

    AMGX_matrix_upload_all(_amgx_matrix_handle, n_local_rows, local_nnz,
                           block_dim_x, block_dim_y, amgx_matrix.row_ptr_dev,
                           amgx_matrix.column_index_dev, amgx_matrix.val_dev,
                           nullptr);
    AMGX_solver_setup(_amgx_solver_handle, _amgx_matrix_handle);
  }
#endif
}

template <typename VectorType>
DirectDeviceOperator<VectorType>::~DirectDeviceOperator()
{
#if MFMG_WITH_AMGX
  if (_solver == "amgx")
  {
    // Clean up the memory
    if (_amgx_solver_handle != nullptr)
    {
      AMGX_solver_destroy(_amgx_solver_handle);
      _amgx_solver_handle = nullptr;
    }

    if (_amgx_solution_handle != nullptr)
    {
      AMGX_vector_destroy(_amgx_solution_handle);
      _amgx_matrix_handle = nullptr;
    }

    if (_amgx_rhs_handle != nullptr)
    {
      AMGX_vector_destroy(_amgx_rhs_handle);
      _amgx_rhs_handle = nullptr;
    }

    if (_amgx_matrix_handle != nullptr)
    {
      AMGX_matrix_destroy(_amgx_matrix_handle);
      _amgx_matrix_handle = nullptr;
    }

    if (_amgx_res_handle != nullptr)
    {
      AMGX_resources_destroy(_amgx_res_handle);
      _amgx_res_handle = nullptr;
    }

    if (_amgx_config_handle != nullptr)
    {
      AMGX_config_destroy(_amgx_config_handle);
      _amgx_config_handle = nullptr;
    }

    AMGX_finalize_plugins();
    AMGX_finalize();
  }
#endif
}

template <typename VectorType>
void DirectDeviceOperator<VectorType>::apply(VectorType const &b,
                                             VectorType &x) const
{
#if MFMG_WITH_AMGX
  if (_solver == "amgx")
  {
    // We need to reorder b because we had to reorder the matrix. So we need to
    // cast the const away
    auto &amgx_b = const_cast<VectorType &>(b);
    internal::DirectOperator<VectorType>::amgx_solve(
        _row_map, _amgx_rhs_handle, _amgx_solution_handle, _amgx_solver_handle,
        _matrix, amgx_b, x);
  }
  else
#endif
    internal::DirectOperator<VectorType>::apply(_cuda_handle, _matrix, _solver,
                                                b, x);
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
} // namespace mfmg

#endif
