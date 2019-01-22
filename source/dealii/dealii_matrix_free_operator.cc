/*************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                           *
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
#include <mfmg/dealii/dealii_matrix_free_mesh_evaluator.hpp>
#include <mfmg/dealii/dealii_matrix_free_operator.hpp>
#include <mfmg/dealii/dealii_trilinos_matrix_operator.hpp>

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
    std::shared_ptr<MeshEvaluator> matrix_free_mesh_evaluator)
    : _mesh_evaluator(std::move(matrix_free_mesh_evaluator))
{
  int const dim = _mesh_evaluator->get_dim();
  std::string const downcasting_failure_error_message =
      "Must pass a matrix free mesh evaluator to create an operator";
  if (dim == 2)
  {
    ASSERT(std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<2>>(
               _mesh_evaluator) != nullptr,
           downcasting_failure_error_message);
  }
  else if (dim == 3)
  {
    ASSERT(std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<3>>(
               _mesh_evaluator) != nullptr,
           downcasting_failure_error_message);
  }
  else
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }
}

template <typename VectorType>
void DealIIMatrixFreeOperator<VectorType>::vmult(VectorType &dst,
                                                 VectorType const &src) const
{
  _mesh_evaluator->get_dim() == 2
      ? std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<2>>(
            _mesh_evaluator)
            ->vmult(dst, src)
      : std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<3>>(
            _mesh_evaluator)
            ->vmult(dst, src);
}

template <typename VectorType>
typename DealIIMatrixFreeOperator<VectorType>::size_type
DealIIMatrixFreeOperator<VectorType>::m() const
{
  return _mesh_evaluator->get_dim() == 2
             ? std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<2>>(
                   _mesh_evaluator)
                   ->m()
             : std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<3>>(
                   _mesh_evaluator)
                   ->m();
}

template <typename VectorType>
typename DealIIMatrixFreeOperator<VectorType>::size_type
DealIIMatrixFreeOperator<VectorType>::n() const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return typename DealIIMatrixFreeOperator<VectorType>::size_type{};
}

template <typename VectorType>
typename DealIIMatrixFreeOperator<VectorType>::value_type
    DealIIMatrixFreeOperator<VectorType>::el(size_type /*i*/,
                                             size_type /*j*/) const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return typename DealIIMatrixFreeOperator<VectorType>::value_type{};
}

template <typename VectorType>
void DealIIMatrixFreeOperator<VectorType>::apply(VectorType const &x,
                                                 VectorType &y,
                                                 OperatorMode mode) const
{
  if (mode != OperatorMode::NO_TRANS)
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }
  this->vmult(y, x);
}

template <typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIMatrixFreeOperator<VectorType>::transpose() const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
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

  auto tmp = this->build_range_vector();
  auto b_mat = downcast_b->get_matrix();

  auto c_mat = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>(
      tmp->locally_owned_elements(), b_mat->locally_owned_range_indices(),
      tmp->get_mpi_communicator());

  matrix_transpose_matrix_multiply(*c_mat, *b_mat, *this);

  return std::make_shared<DealIITrilinosMatrixOperator<VectorType>>(c_mat);
}

template <typename VectorType>
std::shared_ptr<VectorType>
DealIIMatrixFreeOperator<VectorType>::build_domain_vector() const
{
  auto domain_vector =
      this->build_range_vector(); // what could possibly go wrong...
  return domain_vector;
}

template <typename VectorType>
std::shared_ptr<VectorType>
DealIIMatrixFreeOperator<VectorType>::build_range_vector() const
{
  auto range_vector =
      _mesh_evaluator->get_dim() == 2
          ? std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<2>>(
                _mesh_evaluator)
                ->build_range_vector()
          : std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<3>>(
                _mesh_evaluator)
                ->build_range_vector();
  return range_vector;
}

template <typename VectorType>
size_t DealIIMatrixFreeOperator<VectorType>::grid_complexity() const
{
  // FIXME Returns garbage since throwing not implemented was not an option
  return typename DealIIMatrixFreeOperator<VectorType>::value_type{};
}

template <typename VectorType>
size_t DealIIMatrixFreeOperator<VectorType>::operator_complexity() const
{
  // FIXME Returns garbage since throwing not implemented was not an option
  return typename DealIIMatrixFreeOperator<VectorType>::value_type{};
}

template <typename VectorType>
typename DealIIMatrixFreeOperator<VectorType>::vector_type
DealIIMatrixFreeOperator<VectorType>::get_diagonal_inverse() const
{
  auto diagonal_inverse =
      _mesh_evaluator->get_dim() == 2
          ? std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<2>>(
                _mesh_evaluator)
                ->get_diagonal_inverse()
          : std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<3>>(
                _mesh_evaluator)
                ->get_diagonal_inverse();
  return diagonal_inverse;
}
} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_VECTORTYPE(TUPLE(DealIIMatrixFreeOperator))
