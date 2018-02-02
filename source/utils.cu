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

#include <mfmg/utils.cuh>

namespace mfmg
{
SparseMatrixDevice<double>
convert_matrix(dealii::TrilinosWrappers::SparseMatrix const &sparse_matrix)
{
  unsigned int const n_local_rows = sparse_matrix.local_size();
  std::vector<double> val;
  std::vector<int> column_index;
  std::vector<int> row_ptr(n_local_rows + 1);
  unsigned int local_nnz = 0;
  for (unsigned int row = 0; row < n_local_rows; ++row)
  {
    int n_entries;
    double *values;
    int *indices;
    sparse_matrix.trilinos_matrix().ExtractMyRowView(row, n_entries, values,
                                                     indices);

    val.insert(val.end(), values, values + n_entries);
    row_ptr[row + 1] = row_ptr[row] + n_entries;
    // Trilinos does not store the column indices directly
    for (int i = 0; i < n_entries; ++i)
      column_index.push_back(dealii::TrilinosWrappers::global_column_index(
          sparse_matrix.trilinos_matrix(), indices[i]));
    local_nnz += n_entries;
  }

  SparseMatrixDevice<double> sparse_matrix_dev(
      internal::copy_to_gpu(val), internal::copy_to_gpu(column_index),
      internal::copy_to_gpu(row_ptr), local_nnz, n_local_rows);

  return sparse_matrix_dev;
}
}
