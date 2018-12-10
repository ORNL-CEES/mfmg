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

#ifndef MFMG_DEALII_MATRIX_FREE_OPERATOR_HPP
#define MFMG_DEALII_MATRIX_FREE_OPERATOR_HPP

#include <mfmg/common/mesh_evaluator.hpp>
#include <mfmg/common/operator.hpp>

#include <deal.II/base/subscriptor.h>

namespace mfmg
{
template <typename VectorType>
class DealIIMatrixFreeOperator : public Operator<VectorType>,
                                 public dealii::Subscriptor
{
public:
  using vector_type = VectorType;
  using size_type = typename VectorType::size_type;
  using value_type = typename VectorType::value_type;

  DealIIMatrixFreeOperator(
      std::shared_ptr<MeshEvaluator> matrix_free_mesh_evaluator);

  void apply(vector_type const &x, vector_type &y) const override final;

  std::shared_ptr<Operator<VectorType>> transpose() const override final;

  std::shared_ptr<Operator<VectorType>>
  multiply(std::shared_ptr<Operator<VectorType> const> b) const override final;

  std::shared_ptr<Operator<VectorType>> multiply_transpose(
      std::shared_ptr<Operator<VectorType> const> b) const override final;

  void vmult(vector_type &dst, vector_type const &src) const;

  size_type m() const;

  size_type n() const;

  value_type el(size_type i, size_type j) const;

  std::shared_ptr<vector_type> build_domain_vector() const override final;

  std::shared_ptr<vector_type> build_range_vector() const override final;

  size_t grid_complexity() const override final;

  size_t operator_complexity() const override final;

  vector_type get_diagonal_inverse() const;

private:
  std::shared_ptr<MeshEvaluator> _mesh_evaluator;
};
} // namespace mfmg

#endif
