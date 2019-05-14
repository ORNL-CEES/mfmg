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

#include <mfmg/common/exceptions.hpp>
#include <mfmg/dealii/dealii_utils.hpp>

#include <EpetraExt_MultiVectorOut.h>
#include <EpetraExt_RowMatrixOut.h>

namespace mfmg
{
dealii::LinearAlgebra::distributed::Vector<double>
extract_row(dealii::TrilinosWrappers::SparseMatrix const &matrix,
            dealii::types::global_dof_index global_j)
{
  int num_entries = 0;
  double *values = nullptr;
  int *local_indices = nullptr;
  ASSERT(matrix.trilinos_matrix().IndicesAreLocal(), "Indices are not local");
  auto const local_j =
      matrix.locally_owned_range_indices().index_within_set(global_j);
  if (local_j != dealii::numbers::invalid_dof_index)
  {
    auto const error_code = matrix.trilinos_matrix().ExtractMyRowView(
        local_j, num_entries, values, local_indices);
    ASSERT(error_code == 0,
           "Non-zero error code (" + std::to_string(error_code) +
               ") returned by Epetra_CrsMatrix::ExtractMyRowView()");
  }

  dealii::IndexSet ghost_indices(matrix.locally_owned_domain_indices().size());
  for (int k = 0; k < num_entries; ++k)
  {
    ghost_indices.add_index(matrix.trilinos_matrix().GCID(local_indices[k]));
  }
  ghost_indices.compress();
  dealii::LinearAlgebra::distributed::Vector<double> ghosted_vector(
      matrix.locally_owned_domain_indices(), ghost_indices,
      matrix.get_mpi_communicator());
  ghosted_vector = 0.;
  for (int k = 0; k < num_entries; ++k)
  {
    auto const global_index = matrix.trilinos_matrix().GCID(local_indices[k]);
    ghosted_vector[global_index] = values[k];
  }
  ghosted_vector.compress(dealii::VectorOperation::add);

  dealii::LinearAlgebra::distributed::Vector<double> vector(
      matrix.locally_owned_domain_indices(), matrix.get_mpi_communicator());
  vector = ghosted_vector;
  return vector;
}

// TODO: write down 4 maps
void matrix_market_output_file(
    const std::string &filename,
    const dealii::TrilinosWrappers::SparseMatrix &matrix)
{
  int rv = EpetraExt::RowMatrixToMatrixMarketFile(filename.c_str(),
                                                  matrix.trilinos_matrix());
  ASSERT(rv == 0, "EpetraExt::RowMatrixToMatrixMarketFile return value is " +
                      std::to_string(rv));
}

void matrix_market_output_file(const std::string &filename,
                               const dealii::SparseMatrix<double> &matrix)
{
  dealii::TrilinosWrappers::SparseMatrix trilinos_matrix;
  trilinos_matrix.reinit(matrix);

  matrix_market_output_file(filename, trilinos_matrix);
}

// TODO: write the map
void matrix_market_output_file(
    const std::string &filename,
    const dealii::TrilinosWrappers::MPI::Vector &vector)
{
  int rv = EpetraExt::MultiVectorToMatrixMarketFile(filename.c_str(),
                                                    vector.trilinos_vector());
  ASSERT(rv == 0, "EpetraExt::RowMatrixToMatrixMarketFile return value is " +
                      std::to_string(rv));
}
} // namespace mfmg
