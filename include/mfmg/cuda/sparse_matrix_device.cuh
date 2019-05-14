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

#ifndef SPARSE_MATRIX_DEVICE_CUH
#define SPARSE_MATRIX_DEVICE_CUH

#ifdef MFMG_WITH_CUDA

#include <deal.II/base/index_set.h>
#include <deal.II/lac/la_parallel_vector.h>

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

  SparseMatrixDevice(SparseMatrixDevice<ScalarType> const &other);

  SparseMatrixDevice(MPI_Comm comm, ScalarType *val_dev, int *column_index_dev,
                     int *row_ptr_dev, unsigned int local_nnz,
                     dealii::IndexSet const &range_indexset,
                     dealii::IndexSet const &domain_indexset);

  SparseMatrixDevice(MPI_Comm comm, ScalarType *val_dev, int *column_index_dev,
                     int *row_ptr_dev, unsigned int local_nnz,
                     dealii::IndexSet const &range_indexset,
                     dealii::IndexSet const &domain_indexset,
                     cusparseHandle_t cusparse_handle);

  ~SparseMatrixDevice();

  SparseMatrixDevice<ScalarType> &
  operator=(SparseMatrixDevice<ScalarType> &&other);

  /**
   * Reinitialize the matrix. This can only be called if the object it empty.
   */
  void reinit(MPI_Comm comm, ScalarType *val_dev, int *column_index_dev,
              int *row_ptr_dev, unsigned int local_nnz,
              dealii::IndexSet const &range_indexset,
              dealii::IndexSet const &domain_indexset,
              cusparseHandle_t cusparse_handle);

  unsigned int m() const { return _range_indexset.size(); }

  unsigned int n_local_rows() const { return _range_indexset.n_elements(); }

  unsigned int n() const { return _domain_indexset.size(); }

  unsigned int local_nnz() const { return _local_nnz; }

  unsigned int n_nonzero_elements() const { return _nnz; }

  dealii::IndexSet locally_owned_domain_indices() const;

  dealii::IndexSet locally_owned_range_indices() const;

  void vmult(dealii::LinearAlgebra::distributed::Vector<
                 ScalarType, dealii::MemorySpace::CUDA> &dst,
             dealii::LinearAlgebra::distributed::Vector<
                 ScalarType, dealii::MemorySpace::CUDA> const &src) const;

  /**
   * Perform the matrix-matrix multiplication C=A*B. This function assumes that
   * the calling matrix A and B have compatible sizes. The size of C will be set
   * within this function.
   */
  void mmult(SparseMatrixDevice<ScalarType> &C,
             SparseMatrixDevice<ScalarType> const &B) const;

  MPI_Comm get_mpi_communicator() const { return _comm; }

  ScalarType *val_dev;
  int *column_index_dev;
  int *row_ptr_dev;
  cusparseHandle_t cusparse_handle;
  cusparseMatDescr_t descr;

private:
  MPI_Comm _comm;
  unsigned int _local_nnz;
  unsigned int _nnz;
  dealii::IndexSet _range_indexset;
  dealii::IndexSet _domain_indexset;
};
} // namespace mfmg

#endif

#endif
