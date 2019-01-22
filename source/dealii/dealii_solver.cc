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

#include <mfmg/common/instantiation.hpp>
#include <mfmg/common/utils.hpp>
#include <mfmg/dealii/dealii_solver.hpp>
#include <mfmg/dealii/dealii_trilinos_matrix_operator.hpp>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <ml_MultiLevelPreconditioner.h>
#include <ml_Preconditioner.h>

namespace mfmg
{
template <typename VectorType>
DealIISolver<VectorType>::DealIISolver(
    std::shared_ptr<Operator<VectorType> const> op,
    std::shared_ptr<boost::property_tree::ptree const> params)
    : Solver<VectorType>(op, params)
{
  std::string coarse_type = this->_params->get("coarse.type", "direct");

  // We only do smoothing if the operator is a DealIITrilinosMatrixOperator
  auto trilinos_operator =
      std::dynamic_pointer_cast<DealIITrilinosMatrixOperator<VectorType> const>(
          this->_operator);
  auto sparse_matrix = trilinos_operator->get_matrix();
  // Make parameters case-insensitive
  std::string coarse_type_lower = coarse_type;
  std::transform(coarse_type_lower.begin(), coarse_type_lower.end(),
                 coarse_type_lower.begin(), ::tolower);

  if (coarse_type_lower == "direct")
  {
    _solver.reset(new dealii::TrilinosWrappers::SolverDirect(_solver_control));
    _solver->initialize(*sparse_matrix);
  }
  else
  {
    if (coarse_type_lower == "ml")
    {
      auto ml_tree = params->get_child_optional("coarse.params");

      // For now, always set defaults to SA
      Teuchos::ParameterList ml_params;
      ML_Epetra::SetDefaults("SA", ml_params);

      if (ml_tree)
      {
        // Augment with user provided parameters
        ptree2plist(*ml_tree, ml_params);
      }

      _smoother.reset(new dealii::TrilinosWrappers::PreconditionAMG());
      static_cast<dealii::TrilinosWrappers::PreconditionAMG *>(_smoother.get())
          ->initialize(*sparse_matrix, ml_params);
    }
    else
    {
      ASSERT_THROW(false,
                   "Unknown coarse solver name: \"" + coarse_type_lower + "\"");
    }
  }
}

template <typename VectorType>
void DealIISolver<VectorType>::apply(VectorType const &b, VectorType &x) const
{
  if (_solver)
  {
    _solver->solve(x, b);
  }
  else
  {
    _smoother->vmult(x, b);
  }
}
} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_VECTORTYPE(TUPLE(DealIISolver))
