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
#include <mfmg/sparse_matrix_device.cuh>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_index_access.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <tuple>

namespace mfmg
{
namespace internal
{
template <typename ScalarType>
ScalarType *copy_to_gpu(std::vector<ScalarType> const &val)
{
  unsigned int const n_elements = val.size();
  ASSERT(n_elements > 0, "Cannot copy an empty vector to the device");
  ScalarType *val_dev;
  cudaError_t error_code =
      cudaMalloc(&val_dev, n_elements * sizeof(ScalarType));
  ASSERT_CUDA(error_code);
  error_code = cudaMemcpy(val_dev, &val[0], n_elements * sizeof(ScalarType),
                          cudaMemcpyHostToDevice);
  ASSERT_CUDA(error_code);

  return val_dev;
}
}

/**
 * Convert a dealii::SparseMatrix to the regular CSR format and move the data to
 * the GPU.
 */
template <typename ScalarType>
SparseMatrixDevice<ScalarType>
convert_matrix(dealii::SparseMatrix<ScalarType> const &sparse_matrix)
{
  unsigned int const nnz = sparse_matrix.n_nonzero_elements();
  int const n_rows = sparse_matrix.m();
  int const row_ptr_size = n_rows + 1;
  std::vector<ScalarType> val;
  val.reserve(nnz);
  std::vector<int> column_index;
  column_index.reserve(nnz);
  std::vector<int> row_ptr(row_ptr_size, 0);

  // deal.II stores the diagonal first in each row so we need to do some
  // reordering
  for (int row = 0; row < n_rows; ++row)
  {
    auto p_end = sparse_matrix.end(row);
    unsigned int counter = 0;
    for (auto p = sparse_matrix.begin(row); p != p_end; ++p)
    {
      val.emplace_back(p->value());
      column_index.emplace_back(p->column());
      ++counter;
    }
    row_ptr[row + 1] = row_ptr[row] + counter;

    // Sort the elements in the row
    unsigned int const offset = row_ptr[row];
    int const diag_index = column_index[offset];
    ScalarType diag_elem = sparse_matrix.diag_element(row);
    unsigned int pos = 1;
    while ((column_index[offset + pos] < row) && (pos < counter))
    {
      val[offset + pos - 1] = val[offset + pos];
      column_index[offset + pos - 1] = column_index[offset + pos];
      ++pos;
    }
    val[offset + pos - 1] = diag_elem;
    column_index[offset + pos - 1] = diag_index;
  }

  SparseMatrixDevice<ScalarType> sparse_matrix_dev(
      internal::copy_to_gpu(val), internal::copy_to_gpu(column_index),
      internal::copy_to_gpu(row_ptr), nnz, n_rows);

  return sparse_matrix_dev;
}

/**
 * Move a dealii::TrilinosWrappers::SparseMatrix to the GPU.
 */
SparseMatrixDevice<double>
convert_matrix(dealii::TrilinosWrappers::SparseMatrix const &sparse_matrix);

template <typename T>
inline void cuda_free(T *pointer)
{
  cudaError_t cuda_error_code = cudaFree(pointer);
  ASSERT_CUDA(cuda_error_code);
  pointer = nullptr;
}
}

#endif
#endif
