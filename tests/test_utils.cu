/*************************************************************************
 * Copyright (c) 2017 by the mfmg authors                                *
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
  float *val_dev;
  int *column_index_dev;
  int *row_ptr_dev;
  std::tie(val_dev, column_index_dev, row_ptr_dev) =
      mfmg::convert_matrix(sparse_matrix);

  // Copy the matrix from the gpu
  unsigned int const nnz = sparse_matrix.n_nonzero_elements();
  int const n_rows = sparse_matrix.m();
  int const row_ptr_size = n_rows + 1;
  std::vector<float> val_host(nnz);
  cudaError_t error_code = cudaMemcpy(
      &val_host[0], val_dev, nnz * sizeof(float), cudaMemcpyDeviceToHost);
  mfmg::ASSERT_CUDA(error_code);
  std::vector<int> column_index_host(nnz);
  error_code = cudaMemcpy(&column_index_host[0], column_index_dev,
                          nnz * sizeof(int), cudaMemcpyDeviceToHost);
  std::vector<int> row_ptr_host(row_ptr_size);
  error_code = cudaMemcpy(&row_ptr_host[0], row_ptr_dev,
                          row_ptr_size * sizeof(int), cudaMemcpyDeviceToHost);

  // Check the result
  unsigned int pos = 0;
  for (unsigned int i = 0; i < n_rows; ++i)
    for (unsigned int j = row_ptr_host[i]; j < row_ptr_host[i + 1]; ++j, ++pos)
      BOOST_CHECK_EQUAL(val_host[pos], sparse_matrix(i, column_index_host[j]));

  // Free the memiory allocated
  error_code = cudaFree(val_dev);
  mfmg::ASSERT_CUDA(error_code);
  error_code = cudaFree(column_index_dev);
  mfmg::ASSERT_CUDA(error_code);
  error_code = cudaFree(row_ptr_dev);
  mfmg::ASSERT_CUDA(error_code);
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
      double *val_dev;
      int *column_index_dev;
      int *row_ptr_dev;
      std::tie(val_dev, column_index_dev, row_ptr_dev) =
          mfmg::convert_matrix(sparse_matrix);
      // Copy the matrix from the gpu
      int const row_ptr_size = n_local_rows + 1;
      std::vector<double> val_host(nnz);
      cudaError_t error_code = cudaMemcpy(
          &val_host[0], val_dev, nnz * sizeof(double), cudaMemcpyDeviceToHost);
      mfmg::ASSERT_CUDA(error_code);
      std::vector<int> column_index_host(nnz);
      error_code = cudaMemcpy(&column_index_host[0], column_index_dev,
                              nnz * sizeof(int), cudaMemcpyDeviceToHost);
      std::vector<int> row_ptr_host(row_ptr_size);
      error_code =
          cudaMemcpy(&row_ptr_host[0], row_ptr_dev, row_ptr_size * sizeof(int),
                     cudaMemcpyDeviceToHost);

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
