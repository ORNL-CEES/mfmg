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

#ifndef MFMG_DEALII_MATRIX_OPERATOR_HPP
#define MFMG_DEALII_MATRIX_OPERATOR_HPP

#include <mfmg/common/operator.hpp>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

namespace mfmg
{
template <typename VectorType>
class DealIIMatrixOperator : Operator<VectorType>
{
public:
  using operator_type = DealIIMatrixOperator<VectorType>;
  using vector_type = VectorType;

  DealIIMatrixOperator(
      std::shared_ptr<dealii::SparsityPattern> sparsity_pattern,
      std::shared_ptr<dealii::SparseMatrix<typename VectorType::value_type>>
          sparse_matrix);

  void apply(vector_type const &x, vector_type &y) const override final;

  std::shared_ptr<Operator<VectorType>> multiply_transpose(
      std::shared_ptr<Operator<VectorType> const> b) const override final;

  std::shared_ptr<vector_type> build_domain_vector() const override final;

  std::shared_ptr<vector_type> build_range_vector() const override final;

  size_t grid_complexity() const override final;

  size_t operator_complexity() const override final;

private:
  // The sparsity pattern needs to outlive the sparse matrix, so we declare it
  // first.
  std::shared_ptr<dealii::SparsityPattern> _sparsity_pattern;

  std::shared_ptr<dealii::SparseMatrix<typename VectorType::value_type>>
      _sparse_matrix;
};
} // namespace mfmg

#endif
