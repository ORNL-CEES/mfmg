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
            ->apply(src, dst)
      : std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<3>>(
            _mesh_evaluator)
            ->apply(src, dst);
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
