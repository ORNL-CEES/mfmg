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

#define BOOST_TEST_MODULE utils

#include <mfmg/cuda/sparse_matrix_device.cuh>
#include <mfmg/cuda/utils.cuh>

#include <deal.II/lac/la_parallel_vector.h>

#include <random>
#include <set>

#include "main.cc"

BOOST_AUTO_TEST_CASE(serial_mv)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  unsigned int const comm_size = dealii::Utilities::MPI::n_mpi_processes(comm);
  if (comm_size == 1)
  {
    cusparseHandle_t cusparse_handle = nullptr;
    cusparseStatus_t cusparse_error_code;
    cusparse_error_code = cusparseCreate(&cusparse_handle);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);

    // Build the sparse matrix on the host
    unsigned int const size = 10;
    dealii::IndexSet parallel_partitioning(size);
    for (unsigned int i = 0; i < size; ++i)
      parallel_partitioning.add_index(i);
    parallel_partitioning.compress();
    dealii::TrilinosWrappers::SparseMatrix sparse_matrix(parallel_partitioning);

    unsigned int nnz = 0;
    for (unsigned int i = 0; i < size; ++i)
    {
      std::default_random_engine generator(i);
      std::uniform_int_distribution<int> distribution(0, size - 1);
      std::set<int> column_indices;
      for (unsigned int j = 0; j < 5; ++j)
      {
        int column_index = distribution(generator);
        sparse_matrix.set(i, column_index, static_cast<double>(i + j));
        column_indices.insert(column_index);
      }
      nnz += column_indices.size();
    }
    sparse_matrix.compress(dealii::VectorOperation::insert);

    // Move the sparse matrix to the device and change the format to a regular
    // CSR
    mfmg::SparseMatrixDevice<double> sparse_matrix_dev =
        mfmg::convert_matrix(sparse_matrix);
    cusparseMatDescr_t descr;
    cusparse_error_code = cusparseCreateMatDescr(&descr);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    sparse_matrix_dev.descr = descr;
    sparse_matrix_dev.cusparse_handle = cusparse_handle;

    // Build a vector on the host
    dealii::LinearAlgebra::distributed::Vector<double> vector(
        parallel_partitioning, comm);
    unsigned int vector_local_size = vector.local_size();
    for (unsigned int i = 0; i < vector_local_size; ++i)
      vector[i] = i;

    // Move the vector to the device
    mfmg::VectorDevice<double> vector_dev(vector.get_partitioner());
    cudaError_t cuda_error_code;
    cuda_error_code =
        cudaMemcpy(vector_dev.val_dev, vector.begin(),
                   vector_local_size * sizeof(double), cudaMemcpyHostToDevice);
    mfmg::ASSERT_CUDA(cuda_error_code);

    // Perform the matrix-vector multiplication on the host
    dealii::LinearAlgebra::distributed::Vector<double> result(vector);
    sparse_matrix.vmult(result, vector);

    // Perform the matrix-vector multiplication on the host
    mfmg::VectorDevice<double> result_dev(vector.get_partitioner());

    sparse_matrix_dev.vmult(result_dev, vector_dev);

    // Check the result
    std::vector<double> result_host(vector_local_size);
    mfmg::cuda_mem_copy_to_host(result_dev.val_dev, result_host);
    for (unsigned int i = 0; i < vector_local_size; ++i)
      BOOST_CHECK_CLOSE(result[i], result_host[i], 1e-14);

    // Destroy cusparse_handle
    cusparse_error_code = cusparseDestroy(cusparse_handle);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_handle = nullptr;
  }
}

BOOST_AUTO_TEST_CASE(distributed_mv)
{
  // We assume that the user launched as many processes as there are gpus,
  // that each node as the same number of GPUS, and that each node has at  least
  // two GPUs. The reason for the last assumption is to make sure that the test
  // runs on the tester but not on desktop or laptop that have only one GPU.
  int n_devices = 0;
  cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
  mfmg::ASSERT_CUDA(cuda_error_code);

  cusparseHandle_t cusparse_handle = nullptr;
  cusparseStatus_t cusparse_error_code;
  cusparse_error_code = cusparseCreate(&cusparse_handle);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);

  if (n_devices > 1)
  {
    MPI_Comm comm = MPI_COMM_WORLD;
    unsigned int const comm_size =
        dealii::Utilities::MPI::n_mpi_processes(comm);
    unsigned int const rank = dealii::Utilities::MPI::this_mpi_process(comm);

    // Set the device for each process
    int device_id = rank % n_devices;
    cuda_error_code = cudaSetDevice(device_id);

    // Build the sparse matrix on the host
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

    // Move the sparse matrix to the device and change the format to a regular
    // CSR
    mfmg::SparseMatrixDevice<double> sparse_matrix_dev =
        mfmg::convert_matrix(sparse_matrix);
    cusparseMatDescr_t descr;
    cusparse_error_code = cusparseCreateMatDescr(&descr);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    sparse_matrix_dev.descr = descr;
    sparse_matrix_dev.cusparse_handle = cusparse_handle;

    // Build a vector on the host
    dealii::LinearAlgebra::distributed::Vector<double> vector(
        parallel_partitioning, comm);
    unsigned int vector_local_size = vector.local_size();
    for (unsigned int i = 0; i < vector_local_size; ++i)
      vector.local_element(i) = i;

    // Move the vector to the device
    mfmg::VectorDevice<double> vector_dev(vector.get_partitioner());
    cuda_error_code =
        cudaMemcpy(vector_dev.val_dev, vector.begin(),
                   vector_local_size * sizeof(double), cudaMemcpyHostToDevice);
    mfmg::ASSERT_CUDA(cuda_error_code);

    // Perform the matrix-vector multiplication on the host
    dealii::LinearAlgebra::distributed::Vector<double> result(vector);
    sparse_matrix.vmult(result, vector);

    // Perform the matrix-vector multiplication on the host
    mfmg::VectorDevice<double> result_dev(vector.get_partitioner());

    sparse_matrix_dev.vmult(result_dev, vector_dev);

    // Check the result
    std::vector<double> result_host(vector_local_size);
    mfmg::cuda_mem_copy_to_host(result_dev.val_dev, result_host);
    for (unsigned int i = 0; i < vector_local_size; ++i)
      BOOST_CHECK_CLOSE(result.local_element(i), result_host[i], 1e-14);
  }

  // Destroy cusparse_handle
  cusparse_error_code = cusparseDestroy(cusparse_handle);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_handle = nullptr;
}

template <typename ScalarType>
std::tuple<std::vector<ScalarType>, std::vector<int>, std::vector<int>>
copy_sparse_matrix_to_host(
    mfmg::SparseMatrixDevice<ScalarType> const &sparse_matrix_dev)
{
  std::vector<ScalarType> val(sparse_matrix_dev.local_nnz());
  mfmg::cuda_mem_copy_to_host(sparse_matrix_dev.val_dev, val);

  std::vector<int> column_index(sparse_matrix_dev.local_nnz());
  mfmg::cuda_mem_copy_to_host(sparse_matrix_dev.column_index_dev, column_index);

  std::vector<int> row_ptr(sparse_matrix_dev.m() + 1);
  mfmg::cuda_mem_copy_to_host(sparse_matrix_dev.row_ptr_dev, row_ptr);

  return std::make_tuple(val, column_index, row_ptr);
}

BOOST_AUTO_TEST_CASE(mmult)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  unsigned int const comm_size = dealii::Utilities::MPI::n_mpi_processes(comm);
  int n_devices = 0;
  cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
  mfmg::ASSERT_CUDA(cuda_error_code);
  if ((comm_size == 1) || (comm_size == 2) && (n_devices == 2))
  {
    int const rank = dealii::Utilities::MPI::this_mpi_process(comm);
    cuda_error_code = cudaSetDevice(rank);
    mfmg::ASSERT_CUDA(cuda_error_code);

    cusparseHandle_t cusparse_handle = nullptr;
    cusparseStatus_t cusparse_error_code;
    cusparse_error_code = cusparseCreate(&cusparse_handle);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);

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
    dealii::SparseMatrix<double> A(sparsity_pattern);
    dealii::SparseMatrix<double> B(sparsity_pattern);
    for (unsigned int i = 0; i < size; ++i)
      for (unsigned int j = 0; j < size; ++j)
        if (sparsity_pattern.exists(i, j))
        {
          A.set(i, j, static_cast<double>(i + j));
          B.set(i, j, static_cast<double>(i - j));
        }
    dealii::SparsityPattern sparsity_pattern_c;
    dealii::SparseMatrix<double> C(sparsity_pattern_c);
    A.mmult(C, B);

    // Move the sparse matrices to the device and change the format to a regular
    // CSR
    mfmg::SparseMatrixDevice<double> A_dev = mfmg::convert_matrix(A);
    mfmg::SparseMatrixDevice<double> B_dev = mfmg::convert_matrix(B);
    mfmg::SparseMatrixDevice<double> C_dev = mfmg::convert_matrix(B);
    cusparseMatDescr_t A_descr;
    cusparse_error_code = cusparseCreateMatDescr(&A_descr);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    A_dev.descr = A_descr;
    A_dev.cusparse_handle = cusparse_handle;

    cusparseMatDescr_t B_descr;
    cusparse_error_code = cusparseCreateMatDescr(&B_descr);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatType(B_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatIndexBase(B_descr, CUSPARSE_INDEX_BASE_ZERO);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    B_dev.descr = B_descr;
    B_dev.cusparse_handle = cusparse_handle;

    cusparseMatDescr_t C_descr;
    cusparse_error_code = cusparseCreateMatDescr(&C_descr);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatType(C_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatIndexBase(C_descr, CUSPARSE_INDEX_BASE_ZERO);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    C_dev.descr = C_descr;
    C_dev.cusparse_handle = cusparse_handle;
    A_dev.mmult(C_dev, B_dev);

    // Move C_dev to the host
    std::vector<double> val_host;
    std::vector<int> column_index_host;
    std::vector<int> row_ptr_host;
    std::tie(val_host, column_index_host, row_ptr_host) =
        copy_sparse_matrix_to_host(C_dev);

    // Check the result
    unsigned int const n_rows = C_dev.m();
    unsigned int pos = 0;
    for (unsigned int i = 0; i < n_rows; ++i)
      for (unsigned int j = row_ptr_host[i]; j < row_ptr_host[i + 1];
           ++j, ++pos)
        BOOST_CHECK_EQUAL(val_host[pos], C(i, column_index_host[j]));

    // Destroy cusparse_handle
    cusparse_error_code = cusparseDestroy(cusparse_handle);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_handle = nullptr;
  }
}
