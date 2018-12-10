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
#include <mfmg/dealii/dealii_matrix_free_mesh_evaluator.hpp>
#include <mfmg/dealii/dealii_matrix_free_operator.hpp>
#include <mfmg/dealii/dealii_matrix_free_smoother.hpp>
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
    // Downcast to DealIIMatrixFreeMeshEvaluator
    auto dealii_mesh_evaluator =
        std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<dim>>(
            mesh_evaluator);

    this->_global_operator.reset(
        new DealIIMatrixFreeOperator<VectorType>(dealii_mesh_evaluator));
  }

  return this->_global_operator;
}

// copy/paste from DealIIHierarchyHelpers::build_restrictor()
// only change is downcast of global_operator
template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIMatrixFreeHierarchyHelpers<dim, VectorType>::build_restrictor(
    MPI_Comm comm, std::shared_ptr<MeshEvaluator> mesh_evaluator,
    std::shared_ptr<boost::property_tree::ptree const> params)
{
  // Downcast to DealIIMatrixFreeMeshEvaluator
  auto dealii_mesh_evaluator =
      std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<dim>>(
          mesh_evaluator);

  auto eigensolver_params = params->get_child("eigensolver");
  AMGe_host<dim, DealIIMatrixFreeMeshEvaluator<dim>, VectorType> amge(
      comm, dealii_mesh_evaluator->get_dof_handler(),
      eigensolver_params.get("type", "arpack"));

  auto agglomerate_params = params->get_child("agglomeration");
  int n_eigenvectors = eigensolver_params.get("number of eigenvectors", 1);
  double tolerance = eigensolver_params.get("tolerance", 1e-14);

  auto restrictor_matrix =
      std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();

  auto locally_relevant_global_diag = dealii_mesh_evaluator->get_diagonal();

  amge.setup_restrictor(agglomerate_params, n_eigenvectors, tolerance,
                        *dealii_mesh_evaluator, locally_relevant_global_diag,
                        *restrictor_matrix);

  std::shared_ptr<Operator<VectorType>> op(
      new DealIITrilinosMatrixOperator<VectorType>(restrictor_matrix));

  return op;
}

template <int dim, typename VectorType>
std::shared_ptr<Smoother<VectorType>>
DealIIMatrixFreeHierarchyHelpers<dim, VectorType>::build_smoother(
    std::shared_ptr<Operator<VectorType> const> op,
    std::shared_ptr<boost::property_tree::ptree const> params)
{
  return std::make_shared<DealIIMatrixFreeSmoother<VectorType>>(op, params);
}
} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_DIM_VECTORTYPE(TUPLE(DealIIMatrixFreeHierarchyHelpers))
