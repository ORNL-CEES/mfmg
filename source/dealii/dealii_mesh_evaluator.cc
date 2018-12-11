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

#include <deal.II/dofs/dof_tools.h>

#include <algorithm>
#include <random>
#include <vector>

namespace mfmg
{
template <int dim>
DealIIMeshEvaluator<dim>::DealIIMeshEvaluator(
    dealii::DoFHandler<dim> &dof_handler,
    dealii::AffineConstraints<double> &constraints,
    std::string mesh_evaluator_type)
    : _dof_handler(dof_handler), _constraints(constraints),
      _mesh_evaluator_type(std::move(mesh_evaluator_type))
{
  std::vector<std::string> const valid_mesh_evaluator_types = {
      "DealIIMeshEvaluator", "DealIIMatrixFreeMeshEvaluator"};
  ASSERT(std::find(std::begin(valid_mesh_evaluator_types),
                   std::end(valid_mesh_evaluator_types),
                   _mesh_evaluator_type) !=
             std::end(valid_mesh_evaluator_types),
         "mesh_evaluator_type string argument passed to DealIIMeshEvaluator "
         "constructor is not valid");
}

template <int dim>
int DealIIMeshEvaluator<dim>::get_dim() const
{
  return dim;
}

template <int dim>
std::string DealIIMeshEvaluator<dim>::get_mesh_evaluator_type() const
{
  return _mesh_evaluator_type;
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
  {
    x[i] = (!constraints.is_constrained(i) ? distribution(generator) : 0.);
  }
}

template <int dim>
dealii::LinearAlgebra::distributed::Vector<double>
DealIIMeshEvaluator<dim>::get_diagonal()
{
  dealii::TrilinosWrappers::SparseMatrix system_matrix;
  evaluate_global(get_dof_handler(), get_constraints(), system_matrix);
  auto comm = system_matrix.get_mpi_communicator();

  // Extract the diagonal of the system sparse matrix. Each processor gets the
  // locally relevant indices, i.e., owned + ghost
  dealii::IndexSet locally_owned_dofs =
      system_matrix.locally_owned_domain_indices();
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(get_dof_handler(),
                                                  locally_relevant_dofs);
  dealii::LinearAlgebra::distributed::Vector<double> locally_owned_global_diag(
      locally_owned_dofs, comm);
  for (auto const val : locally_owned_dofs)
  {
    locally_owned_global_diag[val] = system_matrix.diag_element(val);
  }
  locally_owned_global_diag.compress(dealii::VectorOperation::insert);

  dealii::LinearAlgebra::distributed::Vector<double>
      locally_relevant_global_diag(locally_owned_dofs, locally_relevant_dofs,
                                   comm);
  locally_relevant_global_diag = locally_owned_global_diag;

  return locally_relevant_global_diag;
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
