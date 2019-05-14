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

#ifndef UTILS_CUH
#define UTILS_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/common/exceptions.hpp>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace mfmg
{
template <typename ScalarType>
class SparseMatrixDevice;

/**
 * Define the size of a block when launching a CUDA kernel.
 */
unsigned int constexpr block_size = 512;

/**
 * Convert a dealii::SparseMatrix to the regular CSR format and move the data to
 * the GPU.
 */
template <typename ScalarType>
SparseMatrixDevice<ScalarType>
convert_matrix(dealii::SparseMatrix<ScalarType> const &sparse_matrix);

/**
 * Move a dealii::TrilinosWrappers::SparseMatrix to the GPU.
 */
SparseMatrixDevice<double>
convert_matrix(dealii::TrilinosWrappers::SparseMatrix const &sparse_matrix);

/**
 * Move an Epetra_CrsMatrix to the GPU.
 */
SparseMatrixDevice<double>
convert_matrix(Epetra_CrsMatrix const &sparse_matrix);

dealii::TrilinosWrappers::SparseMatrix
convert_to_trilinos_matrix(SparseMatrixDevice<double> const &matrix_dev);

std::tuple<std::unordered_map<int, int>, std::unordered_map<int, int>>
csr_to_amgx(std::unordered_set<int> const &rows_sent,
            mfmg::SparseMatrixDevice<double> &matrix_dev);

#if __CUDACC__

template <typename T>
inline void cuda_free(T *&pointer)
{
  cudaError_t cuda_error_code = cudaFree(pointer);
  ASSERT_CUDA(cuda_error_code);
  pointer = nullptr;
}

template <typename T>
inline void cuda_malloc(T *&pointer, unsigned int n_elements)
{
  cudaError_t cuda_error_code = cudaMalloc(&pointer, n_elements * sizeof(T));
  ASSERT_CUDA(cuda_error_code);
}

template <typename T>
inline void cuda_mem_copy_to_host(T const *pointer_dev,
                                  std::vector<T> &vector_host)
{
  cudaError_t cuda_error_code =
      cudaMemcpy(vector_host.data(), pointer_dev,
                 vector_host.size() * sizeof(T), cudaMemcpyDeviceToHost);
  ASSERT_CUDA(cuda_error_code);
}

template <typename T>
inline void cuda_mem_copy_to_dev(std::vector<T> const &vector_host,
                                 T *pointer_dev)
{
  cudaError_t cuda_error_code =
      cudaMemcpy(pointer_dev, vector_host.data(),
                 vector_host.size() * sizeof(T), cudaMemcpyHostToDevice);
  ASSERT_CUDA(cuda_error_code);
}
#endif

void all_gather(MPI_Comm communicator, unsigned int send_count,
                unsigned int *send_buffer, unsigned int recv_count,
                unsigned int *recv_buffer);

void all_gather(MPI_Comm communicator, unsigned int send_count,
                float *send_buffer, unsigned int recv_count,
                float *recv_buffer);

void all_gather_host(MPI_Comm communicator, unsigned int send_count,
                     double *send_buffer, unsigned int recv_count,
                     double *recv_buffer);

void all_gather_dev(MPI_Comm communicator, unsigned int send_count,
                    float *send_buffer, unsigned int recv_count,
                    float *recv_buffer);

void all_gather_dev(MPI_Comm communicator, unsigned int send_count,
                    double *send_buffer, unsigned int recv_count,
                    double *recv_buffer);

dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::CUDA>
copy_from_host(dealii::LinearAlgebra::distributed::Vector<
               double, dealii::MemorySpace::Host> const &vector_host);

dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Host>
copy_from_dev(dealii::LinearAlgebra::distributed::Vector<
              double, dealii::MemorySpace::CUDA> const &vector_host);
} // namespace mfmg

#endif
#endif
