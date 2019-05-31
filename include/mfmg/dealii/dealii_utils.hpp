/**************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                            *
 * All rights reserved.                                                   *
 *                                                                        *
 * This file is part of the mfmg library. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the  *
 * top-level directory                                                    *
 *                                                                        *
 * SPDX-License-Identifier: BSD-3-Clause                                  *
 **************************************************************************/

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
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix>
matrix_transpose_matrix_multiply(
    dealii::IndexSet const &row_index_set,
    dealii::IndexSet const &col_index_set, MPI_Comm const &comm,
    dealii::TrilinosWrappers::SparseMatrix const &B, Operator const &A)
{
  // C = A * B^T
  // C_ij = A_ik * B^T_kj = A_ik * B_jk

  auto C = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>(
      row_index_set, col_index_set, comm);
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
          C->set(i, j, value);
        }
      }
    }
  }
  C->compress(dealii::VectorOperation::insert);

  return C;

  //--------------------------------------------------------------------------//
  //  THE CODE BELOW CAN BE USED TO GREATLY DECREASE THE MEMORY REQUIREMENT   //
  //  BUT THE CODE ONLY WORKS IN SERIAL                                       //
  //--------------------------------------------------------------------------//
  /*
  auto tmp = A.build_range_vector();

  int const global_n_rows = tmp->size();
  int const global_n_columns = B.m(); //< number of rows in B

  dealii::LinearAlgebra::distributed::Vector<double> dst(
      tmp->locally_owned_elements(), tmp->get_mpi_communicator());
  // NOTE: getting an error that the index set is not compressed when calling
  // IndexSet::index_within_set() directly on dst.locally_owned_elements() so
  // ended up making a copy and calling IndexSet::compress() on it.  Not sure
  // it is the best solution but this will do for now.
  auto index_set = dst.locally_owned_elements();
  index_set.compress();

  auto C = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>(
      row_index_set, col_index_set, comm);

  Epetra_CrsMatrix c_mat(Copy, row_index_set.make_trilinos_map(),
                         col_index_set.make_trilinos_map(), 0 ,
                         true);
  {
    std::vector<std::vector<std::pair<int, double>>> this_is_stupid(
        index_set.size());
    int const eigenvectors_per_aggregate = 1;
    int const max_number_of_aggretes_within_one_hop_of_any_fine_dof =
        9; // 27 in 3D. This also assumes that aggregates have at least 2 cells
           // in each direction.
    std::for_each(this_is_stupid.begin(), this_is_stupid.end(), [](auto &v) {
      v.reserve(max_number_of_aggretes_within_one_hop_of_any_fine_dof *
                eigenvectors_per_aggregate);
    });

    int nnz = 0;
    for (int j = 0; j < global_n_columns; ++j)
    {
      auto const src = extract_row(B, j);
      A.apply(src, dst);

      for (int i = 0; i < global_n_rows; ++i)
      {
        auto const local_i = index_set.index_within_set(i);
        if (local_i != dealii::numbers::invalid_dof_index)
        {
          auto const value = dst[i];
          if (std::abs(value) > 1e-14) // is that an appropriate epsilon?
          {
            this_is_stupid[local_i].push_back(std::make_pair(j, value));
            nnz++;
          }
        }
      }
    }
    auto n = this_is_stupid.size();
    assert(n == index_set.size());

    Epetra_IntSerialDenseVector &c_rowptr = c_mat.ExpertExtractIndexOffset();
    Epetra_IntSerialDenseVector &c_colind = c_mat.ExpertExtractIndices();
    double *&c_values = c_mat.ExpertExtractValues();

    c_rowptr.Resize(n + 1);
    c_colind.Resize(nnz);
    delete[] c_values;
    c_values = new double[nnz];

    int *rowptr_aux = c_rowptr.Values();
    int *colind_aux = c_colind.Values();
    double *values_aux = c_values;

    for (unsigned int i = 0; i < index_set.size(); i++)
    {
      auto &row = this_is_stupid[i];
      rowptr_aux[i + 1] = rowptr_aux[i] + row.size();
      for (unsigned int j = 0; j < row.size(); j++)
      {
        colind_aux[rowptr_aux[i] + j] = row[j].first;
        values_aux[rowptr_aux[i] + j] = row[j].second;
      }
    }
  }

  // c_mat.ExpertStaticFillComplete(C->domain_partitioner(),
  // C->range_partitioner());
  // c_mat.ExpertStaticFillComplete(row_index_set.make_trilinos_map(),
  // row_index_set.make_trilinos_map());
  c_mat.ExpertStaticFillComplete(col_index_set.make_trilinos_map(),
                                 row_index_set.make_trilinos_map());

  C->reinit(c_mat);

  return C;
  */
}

void matrix_market_output_file(
    std::string const &filename,
    dealii::TrilinosWrappers::SparseMatrix const &matrix);

void matrix_market_output_file(std::string const &filename,
                               dealii::SparseMatrix<double> const &matrix);

void matrix_market_output_file(
    std::string const &filename,
    dealii::TrilinosWrappers::MPI::Vector const &vector);
} // namespace mfmg

#endif
