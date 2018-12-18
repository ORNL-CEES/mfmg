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

#ifndef MFMG_DEALII_TRILINOS_MATRIX_OPERATOR_HPP
#define MFMG_DEALII_TRILINOS_MATRIX_OPERATOR_HPP

#include <mfmg/common/operator.hpp>

#include <deal.II/lac/trilinos_sparse_matrix.h>

namespace mfmg
{
template <typename VectorType>
class DealIITrilinosMatrixOperator : public Operator<VectorType>
{
public:
  using vector_type = VectorType;

  DealIITrilinosMatrixOperator(
      std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> sparse_matrix);

  void apply(vector_type const &x, vector_type &y,
             OperatorMode mode = OperatorMode::NO_TRANS) const override final;

  std::shared_ptr<Operator<VectorType>> transpose() const override final;

  std::shared_ptr<Operator<VectorType>>
  multiply(std::shared_ptr<Operator<VectorType> const> b) const override;

  std::shared_ptr<Operator<VectorType>> multiply_transpose(
      std::shared_ptr<Operator<VectorType> const> b) const override;

  std::shared_ptr<vector_type> build_domain_vector() const override final;

  std::shared_ptr<vector_type> build_range_vector() const override final;

  size_t grid_complexity() const override final;

  size_t operator_complexity() const override final;

  std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> get_matrix() const;

protected:
  std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> _sparse_matrix;
};
} // namespace mfmg

#endif
