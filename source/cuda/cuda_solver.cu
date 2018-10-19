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

#include <mfmg/cuda/cuda_matrix_operator.cuh>
#include <mfmg/cuda/cuda_solver.cuh>
#include <mfmg/cuda/dealii_operator_device_helpers.cuh>

#include <set>

namespace mfmg
{
namespace
{
template <typename VectorType>
struct DirectSolver
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
void DirectSolver<VectorType>::apply(
    CudaHandle const &cuda_handle,
    SparseMatrixDevice<typename VectorType::value_type> const &matrix,
    std::string const &solver, VectorType const &b, VectorType &x)
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <>
void DirectSolver<VectorDevice<double>>::apply(
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
void DirectSolver<dealii::LinearAlgebra::distributed::Vector<double>>::apply(
    CudaHandle const &cuda_handle, SparseMatrixDevice<double> const &matrix,
    std::string const &solver,
    dealii::LinearAlgebra::distributed::Vector<double> const &b,
    dealii::LinearAlgebra::distributed::Vector<double> &x)
{
  // Copy to the device
  VectorDevice<double> x_dev(x);
  VectorDevice<double> b_dev(b);

  DirectSolver<VectorDevice<double>>::apply(cuda_handle, matrix, solver, b_dev,
                                            x_dev);

  // Move the data to the host
  std::vector<double> x_host(x.local_size());
  cuda_mem_copy_to_host(x_dev.val_dev, x_host);
  std::copy(x_host.begin(), x_host.end(), x.begin());
}

#if MFMG_WITH_AMGX
template <typename VectorType>
void DirectSolver<VectorType>::amgx_solve(
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
void DirectSolver<VectorDevice<double>>::amgx_solve(
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
void DirectSolver<dealii::LinearAlgebra::distributed::Vector<double>>::
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
} // namespace

template <typename VectorType>
CudaSolver<VectorType>::CudaSolver(
    CudaHandle const &cuda_handle,
    std::shared_ptr<Operator<VectorType> const> op,
    std::shared_ptr<boost::property_tree::ptree const> params)
    : Solver<VectorType>(op, params), _cuda_handle(cuda_handle),
      _amgx_config_handle(nullptr), _amgx_res_handle(nullptr),
      _amgx_matrix_handle(nullptr), _amgx_rhs_handle(nullptr),
      _amgx_solution_handle(nullptr), _amgx_solver_handle(nullptr)
{
  // Transform to lower case
  _solver = this->_params->get("solver.type", "lu_dense");
  std::transform(_solver.begin(), _solver.end(), _solver.begin(), tolower);

#if MFMG_WITH_AMGX
  // If the solver is amgx, we need the name of the input file
  if (_solver == "amgx")
  {
    auto amgx_config_file = params->get<std::string>("solver.config_file");

    // Downcast the operator
    auto cuda_operator =
        std::dynamic_pointer_cast<CudaMatrixOperator<VectorType> const>(
            this->_operator);

    // We need to cast away the const because we need to change the format
    // of _matrix to the format supported by amgx
    auto &amgx_matrix = const_cast<SparseMatrixDevice<value_type> &>(
        *(cuda_operator->get_matrix()));

    AMGX_initialize();
    AMGX_initialize_plugins();

    AMGX_config_create_from_file(&_amgx_config_handle, amgx_config_file.data());

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
CudaSolver<VectorType>::~CudaSolver()
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
void CudaSolver<VectorType>::apply(VectorType const &b, VectorType &x) const
{
  // Downcast the operator
  auto cuda_operator =
      std::dynamic_pointer_cast<CudaMatrixOperator<VectorType> const>(
          this->_operator);
  auto matrix = cuda_operator->get_matrix();
#if MFMG_WITH_AMGX
  if (_solver == "amgx")
  {
    // We need to reorder b because we had to reorder the matrix. So we need to
    // cast the const away
    auto &amgx_b = const_cast<VectorType &>(b);
    DirectSolver<VectorType>::amgx_solve(
        _row_map, _amgx_rhs_handle, _amgx_solution_handle, _amgx_solver_handle,
        *matrix, amgx_b, x);
  }
  else
#endif
    DirectSolver<VectorType>::apply(_cuda_handle, *matrix, _solver, b, x);
}
} // namespace mfmg

// Explicit Instantiation
template class mfmg::CudaSolver<mfmg::VectorDevice<double>>;
template class mfmg::CudaSolver<
    dealii::LinearAlgebra::distributed::Vector<double>>;
