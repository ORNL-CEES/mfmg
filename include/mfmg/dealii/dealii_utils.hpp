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

#ifndef MFMG_DEALII_UTILS_H
#define MFMG_DEALII_UTILS_H

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <string>

namespace mfmg
{
dealii::LinearAlgebra::distributed::Vector<double>
extract_row(dealii::TrilinosWrappers::SparseMatrix const &matrix,
            dealii::types::global_dof_index global_j);

// matrix_transpose_matrix_multiply(C, B, A) performs the matrix-matrix
// multiplication with the transpose of B, i.e. C = A * B^T
//
// Note that it is different from deal.II's SparseMatrix::Tmmult(C, B) which
// performs C = A^T * B
template <typename Operator>
void matrix_transpose_matrix_multiply(
    dealii::TrilinosWrappers::SparseMatrix &C,
    dealii::TrilinosWrappers::SparseMatrix const &B, Operator const &A)
{
  // C = A * B^T
  // C_ij = A_ik * B^T_kj = A_ik * B_jk

  auto tmp = A.build_range_vector();
  std::vector<dealii::types::global_dof_index> i_indices;
  tmp->locally_owned_elements().fill_index_vector(i_indices);
  std::vector<dealii::types::global_dof_index> j_indices;
  B.locally_owned_range_indices().fill_index_vector(j_indices);

  int const global_n_rows = tmp->size();
  int const global_n_columns = B.m(); //< number of rows in B
  for (int j = 0; j < global_n_columns; ++j)
  {
    auto const src = extract_row(B, j);

    std::remove_const<decltype(src)>::type dst(tmp->locally_owned_elements(),
                                               tmp->get_mpi_communicator());
    A.apply(src, dst);

    // NOTE: getting an error that the index set is not compressed when calling
    // IndexSet::index_within_set() directly on dst.locally_owned_elements() so
    // ended up making a copy and calling IndexSet::compress() on it.  Not sure
    // it is the best solution but this will do for now.
    auto index_set = dst.locally_owned_elements();
    index_set.compress();
    for (int i = 0; i < global_n_rows; ++i)
    {
      auto const local_i = index_set.index_within_set(i);
      if (local_i != dealii::numbers::invalid_dof_index)
      {
        auto const value = dst[i];
        if (std::abs(value) > 1e-14) // is that an appropriate epsilon?
        {
          C.set(i, j, value);
        }
      }
    }
  }
  C.compress(dealii::VectorOperation::insert);
}

void matrix_market_output_file(
    const std::string &filename,
    const dealii::TrilinosWrappers::SparseMatrix &matrix);

void matrix_market_output_file(
    const std::string &filename,
    const dealii::TrilinosWrappers::MPI::Vector &vector);
} // namespace mfmg

#endif
