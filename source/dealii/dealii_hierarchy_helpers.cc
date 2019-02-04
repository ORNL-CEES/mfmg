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
#include <mfmg/common/operator.hpp>
#include <mfmg/dealii/dealii_hierarchy_helpers.hpp>
#include <mfmg/dealii/dealii_smoother.hpp>
#include <mfmg/dealii/dealii_solver.hpp>
#include <mfmg/dealii/dealii_trilinos_matrix_operator.hpp>
#include <mfmg/dealii/dealii_utils.hpp>

#include <EpetraExt_MatrixMatrix.h>
#include <Epetra_Map.h>

#include <boost/smart_ptr/make_unique.hpp>

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
std::shared_ptr<Operator<VectorType>>
DealIIHierarchyHelpers<dim, VectorType>::build_restrictor(
    MPI_Comm comm, std::shared_ptr<MeshEvaluator> mesh_evaluator,
    std::shared_ptr<boost::property_tree::ptree const> params)
{
  // Downcast to DealIIMeshEvaluator
  auto dealii_mesh_evaluator =
      std::dynamic_pointer_cast<DealIIMeshEvaluator<dim>>(mesh_evaluator);

  auto eigensolver_params = params->get_child("eigensolver");
  int n_eigenvectors = eigensolver_params.get("number of eigenvectors", 1);
  double tolerance = eigensolver_params.get("tolerance", 1e-14);

  auto restrictor_matrix =
      std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();

  auto locally_relevant_global_diag = dealii_mesh_evaluator->get_diagonal();

  bool fast_ap = params->get("fast_ap", false);
  _amge.reset(new AMGe_host<dim, DealIIMeshEvaluator<dim>, VectorType>(
      comm, dealii_mesh_evaluator->get_dof_handler(), eigensolver_params));
  auto agglomerate_params = params->get_child("agglomeration");
  if (fast_ap)
  {
    _eigenvector_matrix.reset(new dealii::TrilinosWrappers::SparseMatrix());
    std::unique_ptr<dealii::TrilinosWrappers::SparseMatrix>
        delta_eigenvector_matrix(new dealii::TrilinosWrappers::SparseMatrix());
    _amge->setup_restrictor(agglomerate_params, n_eigenvectors, tolerance,
                            *dealii_mesh_evaluator,
                            locally_relevant_global_diag, restrictor_matrix,
                            _eigenvector_matrix, delta_eigenvector_matrix);

    // This part should be replaced and done in parallel using WorkStream
    auto system_matrix =
        std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
    // Call user function to fill in the system matrix
    dealii_mesh_evaluator->evaluate_global(
        dealii_mesh_evaluator->get_dof_handler(),
        dealii_mesh_evaluator->get_constraints(), *system_matrix);

    _delta_correction_matrix =
        std::make_unique<dealii::TrilinosWrappers::SparseMatrix>(
            system_matrix->locally_owned_range_indices(),
            delta_eigenvector_matrix->locally_owned_range_indices(),
            system_matrix->get_mpi_communicator());

    DealIITrilinosMatrixOperator<VectorType> system_operator(system_matrix);
    matrix_transpose_matrix_multiply(
        *_delta_correction_matrix, *delta_eigenvector_matrix, system_operator);

    auto tmp = std::make_unique<dealii::TrilinosWrappers::SparseMatrix>(
        system_matrix->locally_owned_range_indices(),
        delta_eigenvector_matrix->locally_owned_range_indices(),
        system_matrix->get_mpi_communicator());
    matrix_transpose_matrix_multiply(*tmp, *_eigenvector_matrix,
                                     system_operator);
    _eigenvector_matrix = std::move(tmp);
  }
  else
  {
    _amge->setup_restrictor(agglomerate_params, n_eigenvectors, tolerance,
                            *dealii_mesh_evaluator,
                            locally_relevant_global_diag, *restrictor_matrix);
    // If we don't use fast_ap, we don't need to keep _amge around. So release
    // the memory.
    _amge.reset();
  }

  std::shared_ptr<Operator<VectorType>> op(
      new DealIITrilinosMatrixOperator<VectorType>(restrictor_matrix));

  return op;
}

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIHierarchyHelpers<dim, VectorType>::fast_multiply_transpose()
{
  Epetra_CrsMatrix *ap = nullptr;

  bool const transpose = true;
  bool const no_transpose = false;
  int error_code = EpetraExt::MatrixMatrix::Add(
      _eigenvector_matrix->trilinos_matrix(), no_transpose, 1.,
      _delta_correction_matrix->trilinos_matrix(), no_transpose, 1., ap);
  // We want to use functions that have been deprecated in deal.II but they
  // won't be removed in the foreseeable
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  Epetra_Map range_map = _eigenvector_matrix->range_partitioner();
  Epetra_Map domain_map = _eigenvector_matrix->domain_partitioner();
#pragma GCC diagnostic pop
  ap->FillComplete(domain_map, range_map);

  ASSERT(error_code == 0, "Problem when adding matrices");

  // Copy the Epetra_CrsMatrix to a dealii::TrilinosWrappers::SparseMatrix
  auto dealii_ap = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
  dealii_ap->reinit(*ap);
  delete ap;

  std::shared_ptr<Operator<VectorType>> ap_operator(
      new DealIITrilinosMatrixOperator<VectorType>(dealii_ap));

  return ap_operator;
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
