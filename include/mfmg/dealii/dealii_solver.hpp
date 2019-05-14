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

#ifndef MFMG_DEALII_SOLVER_HPP
#define MFMG_DEALII_SOLVER_HPP

#include <mfmg/common/solver.hpp>

#include <deal.II/lac/trilinos_solver.h>

namespace mfmg
{
template <typename VectorType>
class DealIISolver final : public Solver<VectorType>
{
public:
  using vector_type = VectorType;

  DealIISolver(std::shared_ptr<Operator<vector_type> const> op,
               std::shared_ptr<boost::property_tree::ptree const> params);

  void apply(vector_type const &b, vector_type &x) const override;

private:
  dealii::SolverControl _solver_control;
  std::unique_ptr<dealii::TrilinosWrappers::SolverDirect> _solver;
  std::unique_ptr<dealii::TrilinosWrappers::PreconditionBase> _smoother;
};
} // namespace mfmg

#endif
