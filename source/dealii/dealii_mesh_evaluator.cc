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

#include <mfmg/common/instantiation.hpp>
#include <mfmg/dealii/dealii_mesh_evaluator.hpp>

namespace mfmg
{
template <int dim>
DealIIMeshEvaluator<dim>::DealIIMeshEvaluator(
    dealii::DoFHandler<dim> &dof_handler,
    dealii::AffineConstraints<double> &constraints)
    : _dof_handler(dof_handler), _constraints(constraints)
{
}

template <int dim>
int DealIIMeshEvaluator<dim>::get_dim() const
{
  return dim;
}

template <int dim>
std::string DealIIMeshEvaluator<dim>::get_mesh_evaluator_type() const
{
  return "DealIIMeshEvaluator";
}

template <int dim>
void DealIIMeshEvaluator<dim>::set_initial_guess(
    dealii::AffineConstraints<double> &constraints,
    dealii::Vector<double> &x) const
{
  unsigned int const n = x.size();

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0., 1.);
  for (unsigned int i = 0; i < n; ++i)
    x[i] = (!constraints.is_constrained(i) ? distribution(generator) : 0.);
}

template <int dim>
dealii::DoFHandler<dim> &DealIIMeshEvaluator<dim>::get_dof_handler()
{
  return _dof_handler;
}

template <int dim>
dealii::AffineConstraints<double> &DealIIMeshEvaluator<dim>::get_constraints()
{
  return _constraints;
}
} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_DIM(TUPLE(DealIIMeshEvaluator))
