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

#define BOOST_TEST_MODULE utils

#include "main.cc"

#include <mfmg/utils.cuh>

#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <algorithm>
#include <random>
#include <set>

template <typename ScalarType>
std::vector<ScalarType> copy_to_host(ScalarType *val_dev,
                                     unsigned int n_elements)
{
  mfmg::ASSERT(n_elements > 0, "Cannot copy an empty array to the host");
  std::vector<ScalarType> val_host(n_elements);
  cudaError_t error_code =
      cudaMemcpy(&val_host[0], val_dev, n_elements * sizeof(ScalarType),
                 cudaMemcpyDeviceToHost);
  mfmg::ASSERT_CUDA(error_code);

  return val_host;
}

template <typename ScalarType>
std::tuple<std::vector<ScalarType>, std::vector<int>, std::vector<int>>
copy_sparse_matrix_to_host(
    mfmg::SparseMatrixDevice<ScalarType> const &sparse_matrix_dev)
{
  std::vector<ScalarType> val =
      copy_to_host(sparse_matrix_dev.val_dev, sparse_matrix_dev.nnz);

  std::vector<int> column_index =
      copy_to_host(sparse_matrix_dev.column_index_dev, sparse_matrix_dev.nnz);

  std::vector<int> row_ptr =
      copy_to_host(sparse_matrix_dev.row_ptr_dev, sparse_matrix_dev.n_rows + 1);

  return std::make_tuple(val, column_index, row_ptr);
}

BOOST_AUTO_TEST_CASE(dealii_sparse_matrix)
{
  // Build the sparsity pattern
  dealii::SparsityPattern sparsity_pattern;
  unsigned int const size = 30;
  std::vector<std::vector<unsigned int>> column_indices(size);
  for (unsigned int i = 0; i < size; ++i)
  {
    std::vector<unsigned int> indices;
    std::default_random_engine generator(i);
    std::uniform_int_distribution<int> distribution(0, size - 1);
    for (unsigned int j = 0; j < 5; ++j)
      indices.push_back(distribution(generator));
    indices.push_back(i);

    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

    column_indices[i] = indices;
  }
  sparsity_pattern.copy_from(size, size, column_indices.begin(),
                             column_indices.end());

  // Build the sparse matrix
  dealii::SparseMatrix<float> sparse_matrix(sparsity_pattern);
  for (unsigned int i = 0; i < size; ++i)
    for (unsigned int j = 0; j < size; ++j)
      if (sparsity_pattern.exists(i, j))
        sparse_matrix.set(i, j, static_cast<float>(i + j));

  // Move the sparse matrix to the device and change the format to a regular CSR
  mfmg::SparseMatrixDevice<float> sparse_matrix_dev =
      mfmg::convert_matrix(sparse_matrix);

  // Copy the matrix from the gpu
  std::vector<float> val_host;
  std::vector<int> column_index_host;
  std::vector<int> row_ptr_host;
  std::tie(val_host, column_index_host, row_ptr_host) =
      copy_sparse_matrix_to_host(sparse_matrix_dev);

  // Check the result
  unsigned int const n_rows = sparse_matrix_dev.n_rows;
  unsigned int pos = 0;
  for (unsigned int i = 0; i < n_rows; ++i)
    for (unsigned int j = row_ptr_host[i]; j < row_ptr_host[i + 1]; ++j, ++pos)
      BOOST_CHECK_EQUAL(val_host[pos], sparse_matrix(i, column_index_host[j]));
}

BOOST_AUTO_TEST_CASE(trilinos_sparse_matrix)
{
  // Build the sparse matrix
  unsigned int const comm_size =
      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  unsigned int const rank =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  unsigned int const n_local_rows = 10;
  unsigned int const row_offset = rank * n_local_rows;
  unsigned int const size = comm_size * n_local_rows;
  dealii::IndexSet parallel_partitioning(size);
  for (unsigned int i = 0; i < n_local_rows; ++i)
    parallel_partitioning.add_index(row_offset + i);
  parallel_partitioning.compress();
  dealii::TrilinosWrappers::SparseMatrix sparse_matrix(parallel_partitioning);

  unsigned int nnz = 0;
  for (unsigned int i = 0; i < n_local_rows; ++i)
  {
    std::default_random_engine generator(i);
    std::uniform_int_distribution<int> distribution(0, size - 1);
    std::set<int> column_indices;
    for (unsigned int j = 0; j < 5; ++j)
    {
      int column_index = distribution(generator);
      sparse_matrix.set(row_offset + i, column_index,
                        static_cast<double>(i + j));
      column_indices.insert(column_index);
    }
    nnz += column_indices.size();
  }
  sparse_matrix.compress(dealii::VectorOperation::insert);

  // Move the sparse matrix to the device. We serialize the access to the GPU so
  // that we don't have any problem when multiple MPI ranks want to access the
  // GPU. In practice, we would need to use MPS but we don't have any control on
  // this (it is the user responsability to set up her GPU correctly).
  for (unsigned int i = 0; i < comm_size; ++i)
  {
    if (i == rank)
    {
      // Move the sparse matrix to the device and change the format to a regular
      // CSR
      mfmg::SparseMatrixDevice<double> sparse_matrix_dev =
          mfmg::convert_matrix(sparse_matrix);

      // Copy the matrix from the gpu
      std::vector<double> val_host;
      std::vector<int> column_index_host;
      std::vector<int> row_ptr_host;
      std::tie(val_host, column_index_host, row_ptr_host) =
          copy_sparse_matrix_to_host(sparse_matrix_dev);

      unsigned int pos = 0;
      for (unsigned int i = 0; i < n_local_rows; ++i)
        for (unsigned int j = row_ptr_host[i]; j < row_ptr_host[i + 1];
             ++j, ++pos)
          BOOST_CHECK_EQUAL(val_host[pos], sparse_matrix(row_offset + i,
                                                         column_index_host[j]));
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

BOOST_AUTO_TEST_CASE(cuda_mpi)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  unsigned int const comm_size = dealii::Utilities::MPI::n_mpi_processes(comm);
  unsigned int const rank = dealii::Utilities::MPI::this_mpi_process(comm);
  int n_devices = 0;
  cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
  mfmg::ASSERT_CUDA(cuda_error_code);

  // Set the device for each process
  int device_id = rank % n_devices;
  cuda_error_code = cudaSetDevice(device_id);

  unsigned int const local_size = 10 + rank;
  std::vector<double> send_buffer_host(local_size, rank);
  double *send_buffer_dev;
  mfmg::cuda_malloc(send_buffer_dev, local_size);
  cuda_error_code =
      cudaMemcpy(send_buffer_dev, send_buffer_host.data(),
                 local_size * sizeof(double), cudaMemcpyHostToDevice);
  mfmg::ASSERT_CUDA(cuda_error_code);

  unsigned int size = 0;
  for (unsigned int i = 0; i < comm_size; ++i)
    size += 10 + i;
  double *recv_buffer_dev;
  mfmg::cuda_malloc(recv_buffer_dev, size);

  mfmg::all_gather_dev(comm, local_size, send_buffer_dev, size,
                       recv_buffer_dev);

  std::vector<double> recv_buffer_host(size);
  cuda_error_code = cudaMemcpy(recv_buffer_host.data(), recv_buffer_dev,
                               size * sizeof(double), cudaMemcpyDeviceToHost);
  mfmg::ASSERT_CUDA(cuda_error_code);

  std::vector<double> recv_buffer_ref;
  recv_buffer_ref.reserve(size);
  for (unsigned int i = 0; i < comm_size; ++i)
    for (unsigned int j = 0; j < 10 + i; ++j)
      recv_buffer_ref.push_back(i);

  for (unsigned int i = 0; i < size; ++i)
    BOOST_CHECK_EQUAL(recv_buffer_host[i], recv_buffer_ref[i]);

  mfmg::cuda_free(send_buffer_dev);
  mfmg::cuda_free(recv_buffer_dev);
}
