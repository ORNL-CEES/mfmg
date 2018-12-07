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
#include <mfmg/dealii/dealii_matrix_operator.hpp>

#include <deal.II/lac/vector.h>

namespace mfmg
{
template <typename VectorType>
DealIIMatrixOperator<VectorType>::DealIIMatrixOperator(
    std::shared_ptr<dealii::SparsityPattern> sparsity_pattern,
    std::shared_ptr<dealii::SparseMatrix<typename VectorType::value_type>>
        sparse_matrix)
    : _sparsity_pattern(std::move(sparsity_pattern)),
      _sparse_matrix(std::move(sparse_matrix))
{
  ASSERT(_sparsity_pattern != nullptr,
         "deal.II matrices require a sparsity pattern");

  ASSERT(_sparse_matrix != nullptr, "The matrix must exist");
}

template <typename VectorType>
void DealIIMatrixOperator<VectorType>::apply(VectorType const &x,
                                             VectorType &y) const
{
  _sparse_matrix->vmult(y, x);
}

template <typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIMatrixOperator<VectorType>::multiply_transpose(
    std::shared_ptr<Operator<VectorType> const> /*b*/) const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
}

template <typename VectorType>
std::shared_ptr<VectorType>
DealIIMatrixOperator<VectorType>::build_domain_vector() const
{
  return std::make_shared<VectorType>(_sparse_matrix->n());
}

template <typename VectorType>
std::shared_ptr<VectorType>
DealIIMatrixOperator<VectorType>::build_range_vector() const
{
  return std::make_shared<VectorType>(_sparse_matrix->m());
}

template <typename VectorType>
size_t DealIIMatrixOperator<VectorType>::grid_complexity() const
{
  return _sparse_matrix->m();
}

template <typename VectorType>
size_t DealIIMatrixOperator<VectorType>::operator_complexity() const
{
  return _sparse_matrix->n_nonzero_elements();
}

} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_SERIALVECTORTYPE(TUPLE(DealIIMatrixOperator))
