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

#include <mfmg/dealii/dealii_trilinos_matrix_operator.hpp>

namespace mfmg
{
template <typename VectorType>
class DealIIMatrixFreeOperator : public DealIITrilinosMatrixOperator<VectorType>
{
public:
  using vector_type = VectorType;

  DealIIMatrixFreeOperator(
      std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> sparse_matrix);

  std::shared_ptr<Operator<VectorType>>
  multiply(std::shared_ptr<Operator<VectorType> const> b) const override final;

  std::shared_ptr<Operator<VectorType>> multiply_transpose(
      std::shared_ptr<Operator<VectorType> const> b) const override final;
};
} // namespace mfmg

#endif
