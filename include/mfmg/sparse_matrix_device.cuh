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
#include <mfmg/vector_device.cuh>

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

  void vmult(VectorDevice<ScalarType> &dst,
             VectorDevice<ScalarType> const &src);

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
