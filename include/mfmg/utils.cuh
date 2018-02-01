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

#ifndef UTILS_CUH
#define UTILS_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/exceptions.hpp>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <tuple>

namespace mfmg
{
template <typename ScalarType>
class SparseMatrixDevice;

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

void all_gather_dev(MPI_Comm communicator, unsigned int send_count,
                    float *send_buffer, unsigned int recv_count,
                    float *recv_buffer);

void all_gather_dev(MPI_Comm communicator, unsigned int send_count,
                    double *send_buffer, unsigned int recv_count,
                    double *recv_buffer);
}

#endif
#endif
