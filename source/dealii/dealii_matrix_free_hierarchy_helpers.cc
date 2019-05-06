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
#include <mfmg/dealii/amge_host.hpp>
#include <mfmg/dealii/dealii_matrix_free_hierarchy_helpers.hpp>
#include <mfmg/dealii/dealii_matrix_free_mesh_evaluator.hpp>
#include <mfmg/dealii/dealii_matrix_free_operator.hpp>
#include <mfmg/dealii/dealii_matrix_free_smoother.hpp>
#include <mfmg/dealii/dealii_solver.hpp>
#include <mfmg/dealii/dealii_trilinos_matrix_operator.hpp>

#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <EpetraExt_MatrixMatrix.h>

#include <unordered_map>

namespace mfmg
{
// copy/paste from DealIIMatrixFreeHierarchyHelpers::get_global_operator()
// only change is _global_operator.reset()
template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIMatrixFreeHierarchyHelpers<dim, VectorType>::get_global_operator(
    std::shared_ptr<MeshEvaluator> mesh_evaluator)
{
  auto matrix_free_mesh_evaluator =
      std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<dim>>(
          mesh_evaluator);
  ASSERT(matrix_free_mesh_evaluator != nullptr, "downcasting failed");
  std::shared_ptr<Operator<VectorType>> global_operator =
      std::make_shared<DealIIMatrixFreeOperator<dim, VectorType>>(
          matrix_free_mesh_evaluator);

  return global_operator;
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

  auto agglomerate_params = params->get_child("agglomeration");
  int n_eigenvectors = eigensolver_params.get("number of eigenvectors", 1);
  double tolerance = eigensolver_params.get("tolerance", 1e-14);

  auto restrictor_matrix =
      std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();

  auto locally_relevant_global_diag = dealii_mesh_evaluator->get_diagonal();

  bool fast_ap = params->get("fast_ap", false);
  if (fast_ap)
  {
    AMGe_host<dim, DealIIMatrixFreeMeshEvaluator<dim>, VectorType> amge(
        comm, dealii_mesh_evaluator->get_dof_handler(), eigensolver_params);
    std::vector<double> eigenvalues;
    // We use unique_ptr because we want to use a special constructor when we
    // fill the matrix, i.e., we don't want to provide a SparsityPattern. All
    // the reinit functions require a SparsityPattern.
    std::unique_ptr<dealii::TrilinosWrappers::SparseMatrix> eigenvector_matrix;
    std::unique_ptr<dealii::TrilinosWrappers::SparseMatrix>
        delta_eigenvector_matrix;
    amge.setup_restrictor(agglomerate_params, n_eigenvectors, tolerance,
                          *dealii_mesh_evaluator, locally_relevant_global_diag,
                          restrictor_matrix, eigenvector_matrix,
                          delta_eigenvector_matrix, eigenvalues);

    dealii::TrilinosWrappers::SparseMatrix delta_correction_matrix(
        eigenvector_matrix->locally_owned_range_indices(),
        eigenvector_matrix->locally_owned_domain_indices(),
        eigenvector_matrix->get_mpi_communicator());

    // Need to apply delta_eigenvector_matrix
    std::vector<std::vector<unsigned int>> interior_agglomerates;
    std::vector<std::vector<unsigned int>> halo_agglomerates;
    std::tie(interior_agglomerates, halo_agglomerates) =
        amge.build_boundary_agglomerates();
    std::unordered_map<std::pair<unsigned int, unsigned int>, double,
                       boost::hash<std::pair<unsigned int, unsigned int>>>
        delta_correction_acc;
    bool is_halo_agglomerate = false;

    // In case there are no patches we own, we still need to construct the
    // restriction operators. We set the number of eigenvalues to zero instead
    // of returning early and duplicating the setup.
    unsigned int const n_local_eigenvectors =
        interior_agglomerates.empty()
            ? 0
            : eigenvector_matrix->locally_owned_range_indices().n_elements() /
                  interior_agglomerates.size();

    for (auto const &agglomerates_vector :
         {interior_agglomerates, halo_agglomerates})
    {
      struct ScratchData
      {
      } scratch_data;
      struct CopyData
      {
        std::unordered_map<std::pair<unsigned int, unsigned int>, double,
                           boost::hash<std::pair<unsigned int, unsigned int>>>
            delta_correction_local_acc;
      } copy_data;

      auto worker =
          [&](const std::vector<std::vector<unsigned int>>::const_iterator
                  &agglomerate_it,
              ScratchData &, CopyData &local_copy_data) {
            local_copy_data.delta_correction_local_acc.clear();

            dealii::Triangulation<dim> agglomerate_triangulation;
            std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
                     typename dealii::DoFHandler<dim>::active_cell_iterator>
                patch_to_global_map;
            amge.build_agglomerate_triangulation(*agglomerate_it,
                                                 agglomerate_triangulation,
                                                 patch_to_global_map);
            if (patch_to_global_map.empty())
            {
              return;
            }

            // Now that we have the triangulation, we can do the evaluation on
            // the agglomerate
            dealii::DoFHandler<dim> agglomerate_dof_handler(
                agglomerate_triangulation);
            agglomerate_dof_handler.distribute_dofs(
                dealii_mesh_evaluator->get_dof_handler().get_fe());

            // Put the result in the matrix
            // Compute the map between the local and the global dof indices.
            std::vector<dealii::types::global_dof_index> dof_indices_map =
                amge.compute_dof_index_map(patch_to_global_map,
                                           agglomerate_dof_handler);
            unsigned int const n_elem = dof_indices_map.size();
            unsigned int const i = agglomerate_it - agglomerates_vector.begin();
            for (unsigned int j = 0; j < n_local_eigenvectors; ++j)
            {
              unsigned int const local_row = i * n_local_eigenvectors + j;
              unsigned int const global_row =
                  eigenvector_matrix->locally_owned_range_indices()
                      .nth_index_in_set(local_row);
              // Get the vector used for the matrix-vector multiplication
              dealii::Vector<ScalarType> delta_eig(n_elem);
              if (is_halo_agglomerate)
              {
                for (unsigned int k = 0; k < n_elem; ++k)
                {
                  delta_eig[k] =
                      delta_eigenvector_matrix->el(global_row,
                                                   dof_indices_map[k]) +
                      eigenvector_matrix->el(global_row, dof_indices_map[k]);
                }
              }
              else
              {
                for (unsigned int k = 0; k < n_elem; ++k)
                {
                  delta_eig[k] = delta_eigenvector_matrix->el(
                      global_row, dof_indices_map[k]);
                }
              }

              // Perform the matrix-vector multiplication
              dealii::Vector<ScalarType> correction(n_elem);
              dealii_mesh_evaluator->matrix_free_evaluate_agglomerate(
                  agglomerate_dof_handler, delta_eig, correction);

              // We would like to fill the delta correction matrix but we can't
              // because we don't know the sparsity pattern. So we accumulate
              // all the values and then fill the matrix using the set()
              // function.
              for (unsigned int k = 0; k < n_elem; ++k)
              {
                local_copy_data.delta_correction_local_acc[std::make_pair(
                    global_row, dof_indices_map[k])] += correction[k];
              }
            }
          };

      auto copier = [&](const CopyData &local_copy_data) {
        for (const auto &local_pair :
             local_copy_data.delta_correction_local_acc)
        {
          delta_correction_acc[local_pair.first] += local_pair.second;
        }
      };

      dealii::WorkStream::run(agglomerates_vector.begin(),
                              agglomerates_vector.end(), worker, copier,
                              scratch_data, copy_data);

      is_halo_agglomerate = true;
    }

    // Fill delta_correction_matrix
    for (auto const &entry : delta_correction_acc)
    {
      delta_correction_matrix.set(entry.first.first, entry.first.second,
                                  entry.second);
    }
    delta_correction_matrix.compress(dealii::VectorOperation::insert);

    // Add the eigenvector matrix and the delta correction matrix to create ap
    Epetra_CrsMatrix *ap = nullptr;
    // We want to use functions that have been deprecated in deal.II but they
    // won't be removed in the foreseeable future
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    Epetra_Map range_map = eigenvector_matrix->domain_partitioner();
    Epetra_Map domain_map = eigenvector_matrix->range_partitioner();
#pragma GCC diagnostic pop

    auto const range_start = eigenvector_matrix->local_range().first;
    auto const range_end = eigenvector_matrix->local_range().second;
    for (unsigned int row = range_start; row < range_end; ++row)
    {
      for (auto column_iterator = eigenvector_matrix->begin(row);
           column_iterator != eigenvector_matrix->end(row); ++column_iterator)
      {
        column_iterator->value() *= eigenvalues.at(row - range_start);
      }
    }
    eigenvector_matrix->compress(dealii::VectorOperation::insert);

    bool const transpose = true;
    int error_code = EpetraExt::MatrixMatrix::Add(
        eigenvector_matrix->trilinos_matrix(), transpose, 1.,
        delta_correction_matrix.trilinos_matrix(), transpose, 1., ap);
    ap->FillComplete(domain_map, range_map);
    ASSERT(error_code == 0, "Problem when adding matrices");

    // Copy the Epetra_CrsMatrix to a dealii::TrilinosWrappers::SparseMatrix
    auto dealii_ap = std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();
    dealii_ap->reinit(*ap);
    delete ap;

    _ap_operator.reset(new DealIITrilinosMatrixOperator<VectorType>(dealii_ap));
  }
  else
  {
    AMGe_host<dim, DealIIMatrixFreeMeshEvaluator<dim>, VectorType> amge(
        comm, dealii_mesh_evaluator->get_dof_handler(), eigensolver_params);

    amge.setup_restrictor(agglomerate_params, n_eigenvectors, tolerance,
                          *dealii_mesh_evaluator, locally_relevant_global_diag,
                          *restrictor_matrix);
  }

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
  return std::make_shared<DealIIMatrixFreeSmoother<dim, VectorType>>(op,
                                                                     params);
}

template <int dim, typename VectorType>
std::shared_ptr<Solver<VectorType>>
DealIIMatrixFreeHierarchyHelpers<dim, VectorType>::build_coarse_solver(
    std::shared_ptr<Operator<VectorType> const> op,
    std::shared_ptr<boost::property_tree::ptree const> params)
{
  return std::make_shared<DealIISolver<VectorType>>(op, params);
}

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIMatrixFreeHierarchyHelpers<dim, VectorType>::fast_multiply_transpose()
{
  return _ap_operator;
}

} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_DIM_VECTORTYPE(TUPLE(DealIIMatrixFreeHierarchyHelpers))
