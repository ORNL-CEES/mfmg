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
#include <mfmg/dealii/dealii_hierarchy_helpers.hpp>
#include <mfmg/dealii/dealii_mesh_evaluator.hpp>
#include <mfmg/dealii/dealii_smoother.hpp>
#include <mfmg/dealii/dealii_solver.hpp>
#include <mfmg/dealii/dealii_trilinos_matrix_operator.hpp>

#include <deal.II/dofs/dof_tools.h>

namespace mfmg
{
template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIHierarchyHelpers<dim, VectorType>::get_global_operator(
    std::shared_ptr<MeshEvaluator> mesh_evaluator)
{
  if (_global_operator == nullptr)
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

    _global_operator.reset(
        new DealIITrilinosMatrixOperator<VectorType>(system_matrix));
  }

  return _global_operator;
}

template <int dim, typename VectorType>
std::shared_ptr<
    dealii::LinearAlgebra::distributed::Vector<typename VectorType::value_type>>
DealIIHierarchyHelpers<dim, VectorType>::get_locally_relevant_global_diag(
    std::shared_ptr<MeshEvaluator> mesh_evaluator)
{
  if (_locally_relevant_global_diag == nullptr)
  {
    auto system_sparse_matrix =
        std::dynamic_pointer_cast<DealIITrilinosMatrixOperator<VectorType>>(
            get_global_operator(mesh_evaluator))
            ->get_matrix();
    auto comm = system_sparse_matrix->get_mpi_communicator();

    auto const &dof_handler =
        std::dynamic_pointer_cast<DealIIMeshEvaluator<dim>>(mesh_evaluator)
            ->get_dof_handler();
    // Extract the diagonal of the system sparse matrix. Each processor gets the
    // locally relevant indices, i.e., owned + ghost
    dealii::IndexSet locally_owned_dofs =
        system_sparse_matrix->locally_owned_domain_indices();
    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                    locally_relevant_dofs);
    dealii::LinearAlgebra::distributed::Vector<typename VectorType::value_type>
        locally_owned_global_diag(locally_owned_dofs, comm);
    for (auto const val : locally_owned_dofs)
      locally_owned_global_diag[val] = system_sparse_matrix->diag_element(val);
    locally_owned_global_diag.compress(dealii::VectorOperation::insert);

    _locally_relevant_global_diag.reset(
        new dealii::LinearAlgebra::distributed::Vector<
            typename VectorType::value_type>(locally_owned_dofs,
                                             locally_relevant_dofs, comm));
    *(_locally_relevant_global_diag) = locally_owned_global_diag;
  }

  return _locally_relevant_global_diag;
}

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIHierarchyHelpers<dim, VectorType>::build_restrictor(
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
DealIIHierarchyHelpers<dim, VectorType>::build_smoother(
    std::shared_ptr<Operator<VectorType> const> op,
    std::shared_ptr<boost::property_tree::ptree const> params)
{
  return std::make_shared<DealIISmoother<VectorType>>(op, params);
}

template <int dim, typename VectorType>
std::shared_ptr<Solver<VectorType>>
DealIIHierarchyHelpers<dim, VectorType>::build_coarse_solver(
    std::shared_ptr<Operator<VectorType> const> op,
    std::shared_ptr<boost::property_tree::ptree const> params)
{
  return std::make_shared<DealIISolver<VectorType>>(op, params);
}
} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_DIM_VECTORTYPE(TUPLE(DealIIHierarchyHelpers))
