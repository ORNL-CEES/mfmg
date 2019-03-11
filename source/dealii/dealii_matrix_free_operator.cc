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
#include <mfmg/dealii/dealii_utils.hpp>

#include <deal.II/lac/la_parallel_vector.h>

namespace mfmg
{
template <int dim, typename VectorType>
DealIIMatrixFreeOperator<dim, VectorType>::DealIIMatrixFreeOperator(
    std::shared_ptr<DealIIMatrixFreeMeshEvaluator<dim>>
        matrix_free_mesh_evaluator)
    : _mesh_evaluator(std::move(matrix_free_mesh_evaluator))
{
}

template <int dim, typename VectorType>
void DealIIMatrixFreeOperator<dim, VectorType>::vmult(
    VectorType &dst, VectorType const &src) const
{
  _mesh_evaluator->matrix_free_evaluate_global(src, dst);
}

template <int dim, typename VectorType>
typename DealIIMatrixFreeOperator<dim, VectorType>::size_type
DealIIMatrixFreeOperator<dim, VectorType>::m() const
{
  return _mesh_evaluator->m();
}

template <int dim, typename VectorType>
typename DealIIMatrixFreeOperator<dim, VectorType>::size_type
DealIIMatrixFreeOperator<dim, VectorType>::n() const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return size_type{};
}

template <int dim, typename VectorType>
typename DealIIMatrixFreeOperator<dim, VectorType>::value_type
    DealIIMatrixFreeOperator<dim, VectorType>::el(size_type /*i*/,
                                                  size_type /*j*/) const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return value_type{};
}

template <int dim, typename VectorType>
void DealIIMatrixFreeOperator<dim, VectorType>::apply(VectorType const &x,
                                                      VectorType &y,
                                                      OperatorMode mode) const
{
  if (mode != OperatorMode::NO_TRANS)
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }
  this->vmult(y, x);
}

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIMatrixFreeOperator<dim, VectorType>::transpose() const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
}

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIMatrixFreeOperator<dim, VectorType>::multiply(
    std::shared_ptr<Operator<VectorType> const> /*b*/) const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
}

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIMatrixFreeOperator<dim, VectorType>::multiply_transpose(
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

template <int dim, typename VectorType>
std::shared_ptr<VectorType>
DealIIMatrixFreeOperator<dim, VectorType>::build_domain_vector() const
{
  auto domain_vector =
      this->build_range_vector(); // what could possibly go wrong...
  return domain_vector;
}

template <int dim, typename VectorType>
std::shared_ptr<VectorType>
DealIIMatrixFreeOperator<dim, VectorType>::build_range_vector() const
{
  return _mesh_evaluator->build_range_vector();
}

template <int dim, typename VectorType>
size_t DealIIMatrixFreeOperator<dim, VectorType>::grid_complexity() const
{
  // FIXME Returns garbage since throwing not implemented was not an option
  return -1;
}

template <int dim, typename VectorType>
size_t DealIIMatrixFreeOperator<dim, VectorType>::operator_complexity() const
{
  // FIXME Returns garbage since throwing not implemented was not an option
  return -1;
}

template <int dim, typename VectorType>
typename DealIIMatrixFreeOperator<dim, VectorType>::vector_type
DealIIMatrixFreeOperator<dim, VectorType>::get_diagonal_inverse() const
{
  return _mesh_evaluator->get_diagonal_inverse();
}
} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_DIM_VECTORTYPE(TUPLE(DealIIMatrixFreeOperator))
