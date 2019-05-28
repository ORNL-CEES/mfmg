/**************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                            *
 * All rights reserved.                                                   *
 *                                                                        *
 * This file is part of the mfmg library. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the  *
 * top-level directory                                                    *
 *                                                                        *
 * SPDX-License-Identifier: BSD-3-Clause                                  *
 *************************************************************************/

#ifndef MFMG_DEALII_MATRIX_FREE_SMOOTHER_HPP
#define MFMG_DEALII_MATRIX_FREE_SMOOTHER_HPP

#include <mfmg/common/smoother.hpp>
#include <mfmg/dealii/dealii_matrix_free_operator.hpp>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/precondition.h>

#include <boost/property_tree/ptree.hpp>

#include <memory>

namespace mfmg
{
template <int dim, typename VectorType>
class DealIIMatrixFreeSmoother final : public Smoother<VectorType>
{
public:
  using vector_type = VectorType;
  using operator_type = DealIIMatrixFreeOperator<dim, VectorType>;
  using preconditioner_type = dealii::DiagonalMatrix<VectorType>;
  using chebyshev_preconditioner =
      dealii::PreconditionChebyshev<operator_type, vector_type,
                                    preconditioner_type>;

  DealIIMatrixFreeSmoother(
      std::shared_ptr<Operator<vector_type> const> op,
      std::shared_ptr<boost::property_tree::ptree const> params);

  virtual ~DealIIMatrixFreeSmoother() override = default;

  void apply(vector_type const &b, vector_type &x) const override;

private:
  std::unique_ptr<chebyshev_preconditioner> _smoother;
};
} // namespace mfmg

#endif
