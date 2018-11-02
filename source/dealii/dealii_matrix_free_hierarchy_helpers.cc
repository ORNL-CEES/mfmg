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
#include <mfmg/common/operator.hpp>
#include <mfmg/dealii/amge_host.hpp>
#include <mfmg/dealii/dealii_matrix_free_hierarchy_helpers.hpp>
#include <mfmg/dealii/dealii_matrix_free_operator.hpp>
#include <mfmg/dealii/dealii_mesh_evaluator.hpp>
#include <mfmg/dealii/dealii_smoother.hpp>
#include <mfmg/dealii/dealii_trilinos_matrix_operator.hpp>

namespace mfmg
{
// copy/paste from DealIIMatrixFreeHierarchyHelpers::get_global_operator()
// only change is _global_operator.reset()
template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIMatrixFreeHierarchyHelpers<dim, VectorType>::get_global_operator(
    std::shared_ptr<MeshEvaluator> mesh_evaluator)
{
  if (this->_global_operator == nullptr)
  {
    // Downcast to DealIIMeshEvaluator
    auto dealii_mesh_evaluator =
        std::dynamic_pointer_cast<DealIIMeshEvaluator<dim>>(mesh_evaluator);

    auto system_matrix =
        std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();

    // Call user function to fill in the system matrix
    dealii_mesh_evaluator->evaluate_global(
        dealii_mesh_evaluator->get_dof_handler(),
        dealii_mesh_evaluator->get_constraints(), *system_matrix);

    this->_global_operator.reset(
        new DealIIMatrixFreeOperator<VectorType>(system_matrix));
  }

  return this->_global_operator;
}

// copy/paste from DealIIHierarchyHelpers::build_restrictor()
template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIMatrixFreeHierarchyHelpers<dim, VectorType>::build_restrictor(
    MPI_Comm comm, std::shared_ptr<MeshEvaluator> mesh_evaluator,
    std::shared_ptr<boost::property_tree::ptree const> params)
{
  // Downcast to DealIIMeshEvaluator
  auto dealii_mesh_evaluator =
      std::dynamic_pointer_cast<DealIIMeshEvaluator<dim>>(mesh_evaluator);

  auto eigensolver_params = params->get_child("eigensolver");
  AMGe_host<dim, DealIIMeshEvaluator<dim>, VectorType> amge(
      comm, dealii_mesh_evaluator->get_dof_handler(),
      eigensolver_params.get("type", "arpack"));

  auto agglomerate_params = params->get_child("agglomeration");
  int n_eigenvectors = eigensolver_params.get("number of eigenvectors", 1);
  double tolerance = eigensolver_params.get("tolerance", 1e-14);

  auto restrictor_matrix =
      std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
  auto global_operator = get_global_operator(mesh_evaluator);
  auto matrix_free_global_operator =
      std::dynamic_pointer_cast<DealIIMatrixFreeOperator<VectorType>>(
          global_operator);
  // ughhh
  auto system_sparse_matrix = matrix_free_global_operator->get_matrix();
  // why do we pass system matrix to amge?
  amge.setup_restrictor(agglomerate_params, n_eigenvectors, tolerance,
                        *dealii_mesh_evaluator, *system_sparse_matrix,
                        *restrictor_matrix);

  std::shared_ptr<Operator<VectorType>> op(
      new DealIITrilinosMatrixOperator<VectorType>(restrictor_matrix));

  return op;
}
} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_DIM_VECTORTYPE(TUPLE(DealIIMatrixFreeHierarchyHelpers))
