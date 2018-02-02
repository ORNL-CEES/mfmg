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

#ifndef SPARSE_MATRIX_DEVICE_CUH
#define SPARSE_MATRIX_DEVICE_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/exceptions.hpp>

namespace mfmg
{
/**
 * This structure encapsulates the pointers that define a matrix on the device.
 * The destructor frees the allocated memory.
 */
template <typename ScalarType>
struct SparseMatrixDevice
{
  SparseMatrixDevice()
      : val_dev(nullptr), column_index_dev(nullptr), row_ptr_dev(nullptr),
        nnz(0), n_rows(0)
  {
  }

  SparseMatrixDevice(SparseMatrixDevice<ScalarType> &&other)
      : val_dev(other.val_dev), column_index_dev(other.column_index_dev),
        row_ptr_dev(other.row_ptr_dev), nnz(other.nnz), n_rows(other.n_rows)
  {
    other.val_dev = nullptr;
    other.column_index_dev = nullptr;
    other.row_ptr_dev = nullptr;

    other.nnz = 0;
    other.n_rows = 0;
  }

  SparseMatrixDevice(ScalarType *val_dev, int *column_index_dev,
                     int *row_ptr_dev, unsigned int nnz, unsigned int n_rows)
      : val_dev(val_dev), column_index_dev(column_index_dev),
        row_ptr_dev(row_ptr_dev), nnz(nnz), n_rows(n_rows)
  {
  }

  ~SparseMatrixDevice()
  {
    if (val_dev != nullptr)
    {
      cudaError_t error_code = cudaFree(val_dev);
      mfmg::ASSERT_CUDA(error_code);
      val_dev = nullptr;
    }
    if (column_index_dev != nullptr)
    {
      cudaError_t error_code = cudaFree(column_index_dev);
      mfmg::ASSERT_CUDA(error_code);
      column_index_dev = nullptr;
    }
    if (row_ptr_dev != nullptr)
    {
      cudaError_t error_code = cudaFree(row_ptr_dev);
      mfmg::ASSERT_CUDA(error_code);
      row_ptr_dev = nullptr;
    }
  }

  ScalarType *val_dev;
  int *column_index_dev;
  int *row_ptr_dev;

  unsigned int nnz;
  unsigned int n_rows;
};
}

#endif

#endif
