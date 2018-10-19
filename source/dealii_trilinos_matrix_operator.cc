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

#include <mfmg/dealii_trilinos_matrix_operator.hpp>
#include <mfmg/exceptions.hpp>
#include <mfmg/instantiation.hpp>

#include <EpetraExt_MatrixMatrix.h>
#include <EpetraExt_Transpose_RowMatrix.h>

namespace mfmg
{
template <typename VectorType>
DealIITrilinosMatrixOperator<VectorType>::DealIITrilinosMatrixOperator(
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> sparse_matrix)
    : _sparse_matrix(sparse_matrix)
{
}

template <typename VectorType>
void DealIITrilinosMatrixOperator<VectorType>::apply(VectorType const &x,
                                                     VectorType &y) const
{
  _sparse_matrix->vmult(y, x);
}

template <typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIITrilinosMatrixOperator<VectorType>::transpose() const
{
  auto epetra_matrix = _sparse_matrix->trilinos_matrix();

  EpetraExt::RowMatrix_Transpose transposer;
  auto transposed_epetra_matrix =
      dynamic_cast<Epetra_CrsMatrix &>(transposer(epetra_matrix));

  auto transposed_matrix =
      std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
  transposed_matrix->reinit(transposed_epetra_matrix);

  return std::make_shared<DealIITrilinosMatrixOperator<VectorType>>(
      transposed_matrix);
}

template <typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIITrilinosMatrixOperator<VectorType>::multiply(
    std::shared_ptr<Operator<VectorType> const> b) const
{
  // Downcast to TrilinosMatrixOperator
  auto downcast_b =
      std::dynamic_pointer_cast<DealIITrilinosMatrixOperator<VectorType> const>(
          b);

  auto a_mat = this->get_matrix();
  auto b_mat = downcast_b->get_matrix();

  auto c_mat = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
  a_mat->mmult(*c_mat, *b_mat);

  return std::make_shared<DealIITrilinosMatrixOperator<VectorType>>(c_mat);
}

template <typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIITrilinosMatrixOperator<VectorType>::multiply_transpose(
    std::shared_ptr<Operator<VectorType> const> operator_b) const
{
  // Downcast to TrilinosMatrixOperator
  auto downcast_operator_b =
      std::dynamic_pointer_cast<DealIITrilinosMatrixOperator<VectorType> const>(
          operator_b);
  auto a = this->get_matrix();
  auto b = downcast_operator_b->get_matrix();
  auto c = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>(
      a->locally_owned_range_indices(), b->locally_owned_range_indices(),
      a->get_mpi_communicator());
  int error_code = EpetraExt::MatrixMatrix::Multiply(
      a->trilinos_matrix(), false, b->trilinos_matrix(), true,
      const_cast<Epetra_CrsMatrix &>(c->trilinos_matrix()));
  ASSERT(error_code == 0, "EpetraExt::MatrixMatrix::Multiply() returned "
                          "non-zero error code in "
                          "DealIITrilinosMatrixOperator::multiply_transpose()");

  return std::make_shared<DealIITrilinosMatrixOperator<VectorType>>(c);
}

template <typename VectorType>
std::shared_ptr<VectorType>
DealIITrilinosMatrixOperator<VectorType>::build_domain_vector() const
{
  return std::make_shared<vector_type>(
      _sparse_matrix->locally_owned_domain_indices(),
      _sparse_matrix->get_mpi_communicator());
}

template <typename VectorType>
std::shared_ptr<VectorType>
DealIITrilinosMatrixOperator<VectorType>::build_range_vector() const
{
  return std::make_shared<vector_type>(
      _sparse_matrix->locally_owned_range_indices(),
      _sparse_matrix->get_mpi_communicator());
}

template <typename VectorType>
size_t DealIITrilinosMatrixOperator<VectorType>::grid_complexity() const
{
  return _sparse_matrix->m();
}

template <typename VectorType>
size_t DealIITrilinosMatrixOperator<VectorType>::operator_complexity() const
{
  return _sparse_matrix->n_nonzero_elements();
}

template <typename VectorType>
std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix>
DealIITrilinosMatrixOperator<VectorType>::get_matrix() const
{
  return _sparse_matrix;
}
} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_VECTORTYPE(TUPLE(DealIITrilinosMatrixOperator))
