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
#include <mfmg/dealii/dealii_matrix_free_mesh_evaluator.hpp>

namespace mfmg
{
template <int dim>
DealIIMatrixFreeMeshEvaluator<dim>::DealIIMatrixFreeMeshEvaluator(
    dealii::DoFHandler<dim> &dof_handler,
    dealii::AffineConstraints<double> &constraints)
    : DealIIMeshEvaluator<dim>{dof_handler, constraints}
{
}

template <int dim>
std::string DealIIMatrixFreeMeshEvaluator<dim>::get_mesh_evaluator_type() const
{
  return "DealIIMatrixFreeMeshEvaluator";
}

template <int dim>
std::shared_ptr<dealii::LinearAlgebra::distributed::Vector<double>>
DealIIMatrixFreeMeshEvaluator<dim>::build_range_vector() const
{
  // Get the MPI communicator from the dof handler
  auto const &triangulation = (this->_dof_handler).get_triangulation();
  using Triangulation =
      typename std::remove_reference<decltype(triangulation)>::type;
  auto comm = static_cast<dealii::parallel::Triangulation<
      Triangulation::dimension, Triangulation::space_dimension> const &>(
                  triangulation)
                  .get_communicator();

  // Get the set of locally owned DoFs
  auto const &locally_owned_dofs = (this->_dof_handler).locally_owned_dofs();

  return std::make_shared<dealii::LinearAlgebra::distributed::Vector<double>>(
      locally_owned_dofs, comm);
}

template <int dim>
dealii::types::global_dof_index DealIIMatrixFreeMeshEvaluator<dim>::m() const
{
  return (this->_dof_handler).n_dofs();
}
} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_DIM(TUPLE(DealIIMatrixFreeMeshEvaluator))
