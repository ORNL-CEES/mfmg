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

#include <cusparse.h>

namespace mfmg
{
/**
 * This class defines a matrix on the device. The destructor frees the allocated
 * memory.
 */
template <typename ScalarType>
class SparseMatrixDevice
{
public:
  SparseMatrixDevice();

  SparseMatrixDevice(SparseMatrixDevice<ScalarType> &&other);

  SparseMatrixDevice(ScalarType *val_dev, int *column_index_dev,
                     int *row_ptr_dev, unsigned int nnz, unsigned int n_rows);

  SparseMatrixDevice(ScalarType *val_dev, int *column_index_dev,
                     int *row_ptr_dev, cusparseHandle_t cusparse_handle,
                     unsigned int nnz, unsigned int n_rows);

  ~SparseMatrixDevice();

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
  cusparseHandle_t cusparse_handle;
  cusparseMatDescr_t descr;

  unsigned int nnz;
  unsigned int n_rows;
};
}

#endif

#endif
