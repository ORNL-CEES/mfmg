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

#include <deal.II/dofs/dof_accessor.h>

#include <EpetraExt_MatrixMatrix.h>
#include <Epetra_Map.h>

#include <boost/container_hash/hash.hpp>
#include <boost/smart_ptr/make_unique.hpp>

#include <unordered_map>

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
  // TODO make it work with MPI
  if (fast_ap)
  {
    std::vector<double> eigenvalues;
    _eigenvector_matrix.reset(new dealii::TrilinosWrappers::SparseMatrix());
    std::unique_ptr<dealii::TrilinosWrappers::SparseMatrix>
        delta_eigenvector_matrix(new dealii::TrilinosWrappers::SparseMatrix());
    _amge->setup_restrictor(
        agglomerate_params, n_eigenvectors, tolerance, *dealii_mesh_evaluator,
        locally_relevant_global_diag, restrictor_matrix, _eigenvector_matrix,
        delta_eigenvector_matrix, eigenvalues);

    _delta_correction_matrix =
        std::make_unique<dealii::TrilinosWrappers::SparseMatrix>(
            _eigenvector_matrix->locally_owned_range_indices(),
            _eigenvector_matrix->locally_owned_domain_indices(),
            _eigenvector_matrix->get_mpi_communicator());

    // Need to apply delta_eigenvector_matrix
    std::vector<std::vector<unsigned int>> interior_agglomerates;
    std::vector<std::vector<unsigned int>> halo_agglomerates;
    std::tie(interior_agglomerates, halo_agglomerates) =
        _amge->build_boundary_agglomerates();
    std::unordered_map<std::pair<unsigned int, unsigned int>, double,
                       boost::hash<std::pair<unsigned int, unsigned int>>>
        delta_correction_acc;
    bool is_halo_agglomerate = false;
    unsigned int const n_local_eigenvectors =
        _delta_correction_matrix->m() / interior_agglomerates.size();
    for (auto const &agglomerates_vector :
         {interior_agglomerates, halo_agglomerates})
    {
      // TODO use WorkStream
      unsigned int const n_agglomerates = agglomerates_vector.size();
      for (unsigned int i = 0; i < n_agglomerates; ++i)
      {
        auto agglomerate = agglomerates_vector[i];
        dealii::Triangulation<dim> agglomerate_triangulation;
        std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
                 typename dealii::DoFHandler<dim>::active_cell_iterator>
            patch_to_global_map;
        _amge->build_agglomerate_triangulation(
            agglomerate, agglomerate_triangulation, patch_to_global_map);

        // Now that we have the triangulation, we can do the evaluation on the
        // agglomerate
        dealii::DoFHandler<dim> agglomerate_dof_handler(
            agglomerate_triangulation);
        dealii::AffineConstraints<double> agglomerate_constraints;
        dealii::SparsityPattern agglomerate_sparsity_pattern;
        dealii::SparseMatrix<ScalarType> agglomerate_system_matrix;
        // Call user function to build the system matrix
        dealii_mesh_evaluator->evaluate_agglomerate(
            agglomerate_dof_handler, agglomerate_constraints,
            agglomerate_sparsity_pattern, agglomerate_system_matrix);

        // Put the result in the matrix
        // Compute the map between the local and the global dof indices.
        std::vector<dealii::types::global_dof_index> dof_indices_map =
            _amge->compute_dof_index_map(patch_to_global_map,
                                         agglomerate_dof_handler);
        unsigned int const n_elem = dof_indices_map.size();
        for (unsigned int j = 0; j < n_local_eigenvectors; ++j)
        {
          unsigned int const row = i * n_local_eigenvectors + j;
          // Get the vector used for the matrix-vector multiplication
          dealii::Vector<ScalarType> delta_eig(n_elem);
          if (is_halo_agglomerate)
          {
            for (unsigned int k = 0; k < n_elem; ++k)
            {
              delta_eig[k] =
                  delta_eigenvector_matrix->el(row, dof_indices_map[k]) +
                  _eigenvector_matrix->el(row, dof_indices_map[k]) /
                      eigenvalues[row];
            }
          }
          else
          {
            for (unsigned int k = 0; k < n_elem; ++k)
            {
              delta_eig[k] =
                  delta_eigenvector_matrix->el(row, dof_indices_map[k]);
            }
          }

          // Perform the matrix-vector multiplication
          dealii::Vector<ScalarType> correction(n_elem);
          agglomerate_system_matrix.vmult(correction, delta_eig);

          // We would like to fill the delta correction matrix but we can't
          // because we don't know the sparsity pattern. So we accumulate all
          // the values and then fill the matrix using the set() function.
          for (unsigned int k = 0; k < n_elem; ++k)
            delta_correction_acc[std::make_pair(row, dof_indices_map[k])] +=
                correction[k];
        }
      }

      is_halo_agglomerate = true;
    }

    // Fill _delta_correction_matrix
    for (auto const &entry : delta_correction_acc)
      _delta_correction_matrix->set(entry.first.first, entry.first.second,
                                    entry.second);
    _delta_correction_matrix->compress(dealii::VectorOperation::insert);
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
  // We want to use functions that have been deprecated in deal.II but they
  // won't be removed in the foreseeable
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  Epetra_Map range_map = _eigenvector_matrix->domain_partitioner();
  Epetra_Map domain_map = _eigenvector_matrix->range_partitioner();
#pragma GCC diagnostic pop

  bool const transpose = true;
  int error_code = EpetraExt::MatrixMatrix::Add(
      _eigenvector_matrix->trilinos_matrix(), transpose, 1.,
      _delta_correction_matrix->trilinos_matrix(), transpose, 1., ap);
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
