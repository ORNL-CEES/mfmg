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

#include <mfmg/common/exceptions.hpp>
#include <mfmg/common/instantiation.hpp>
#include <mfmg/dealii/dealii_matrix_free_operator.hpp>

#include <deal.II/lac/la_parallel_vector.h>

namespace mfmg
{
namespace
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

  std::vector<dealii::types::global_dof_index> i_indices;
  A.locally_owned_range_indices().fill_index_vector(i_indices);
  std::vector<dealii::types::global_dof_index> j_indices;
  B.locally_owned_range_indices().fill_index_vector(j_indices);

  int const global_n_rows = A.row_partitioner().NumGlobalElements();
  int const global_n_columns = B.row_partitioner().NumGlobalElements();
  for (int j = 0; j < global_n_columns; ++j)
  {
    auto const src = extract_row(B, j);

    std::remove_const<decltype(src)>::type dst(A.locally_owned_range_indices(),
                                               A.get_mpi_communicator());
    A.vmult(dst, src);

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
} // namespace

template <typename VectorType>
DealIIMatrixFreeOperator<VectorType>::DealIIMatrixFreeOperator(
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> sparse_matrix)
    : DealIITrilinosMatrixOperator<VectorType>(sparse_matrix)
{
}

template <typename VectorType>
void DealIIMatrixFreeOperator<VectorType>::vmult(VectorType &dst,
                                                 VectorType const &src) const
{
  (this->_sparse_matrix)->vmult(dst, src);
}

template <typename VectorType>
typename DealIIMatrixFreeOperator<VectorType>::size_type
DealIIMatrixFreeOperator<VectorType>::m() const
{
  return (this->_sparse_matrix)->m();
}

template <typename VectorType>
typename DealIIMatrixFreeOperator<VectorType>::size_type
DealIIMatrixFreeOperator<VectorType>::n() const
{
  return (this->_sparse_matrix)->n();
}

template <typename VectorType>
typename DealIIMatrixFreeOperator<VectorType>::value_type
DealIIMatrixFreeOperator<VectorType>::el(size_type i, size_type j) const
{
  ASSERT(i == j, "was intended for accessing diagonal elements only");
  return (this->_sparse_matrix)->el(i, i);
}
template <typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIMatrixFreeOperator<VectorType>::multiply(
    std::shared_ptr<Operator<VectorType> const> /*b*/) const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
}

template <typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIMatrixFreeOperator<VectorType>::multiply_transpose(
    std::shared_ptr<Operator<VectorType> const> b) const
{
  // Downcast to TrilinosMatrixOperator
  auto downcast_b =
      std::dynamic_pointer_cast<DealIITrilinosMatrixOperator<VectorType> const>(
          b);

  auto a_mat = this->get_matrix();
  auto b_mat = downcast_b->get_matrix();

  auto c_mat = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>(
      a_mat->locally_owned_range_indices(),
      b_mat->locally_owned_range_indices(), a_mat->get_mpi_communicator());

  matrix_transpose_matrix_multiply(*c_mat, *b_mat, *a_mat);

  return std::make_shared<DealIIMatrixFreeOperator<VectorType>>(c_mat);
}
} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_VECTORTYPE(TUPLE(DealIIMatrixFreeOperator))
