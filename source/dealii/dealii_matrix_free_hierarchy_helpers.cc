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

#include <boost/range/combine.hpp>

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
  if (this->_global_operator == nullptr)
  {
    auto matrix_free_mesh_evaluator =
        std::dynamic_pointer_cast<DealIIMatrixFreeMeshEvaluator<dim>>(
            mesh_evaluator);
    ASSERT(matrix_free_mesh_evaluator != nullptr, "downcasting failed");
    this->_global_operator.reset(new DealIIMatrixFreeOperator<dim, VectorType>(
        matrix_free_mesh_evaluator));
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

  auto agglomerate_params = params->get_child("agglomeration");
  int n_eigenvectors = eigensolver_params.get("number of eigenvectors", 1);
  double tolerance = eigensolver_params.get("tolerance", 1e-14);

  auto restrictor_matrix =
      std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();

  auto locally_relevant_global_diag = dealii_mesh_evaluator->get_diagonal();

  bool fast_ap = params->get("fast_ap", false);
  if (fast_ap)
  {
    // TODO make it work with MPI
    ASSERT(dealii::Utilities::MPI::n_mpi_processes(comm) == 1,
           "fast_ap only works in serial");

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
    unsigned int const n_local_eigenvectors =
        delta_correction_matrix.m() / interior_agglomerates.size();
    ASSERT(interior_agglomerates.size() == halo_agglomerates.size(),
           "Every interior agglomerate should correspond to exactly one halo "
           "agglomerate!");
    {
      const auto combined_agglomerate_range =
          boost::combine(interior_agglomerates, halo_agglomerates);

      struct ScratchData
      {
      } scratch_data;
      struct CopyData
      {
        std::unordered_map<std::pair<unsigned int, unsigned int>, double,
                           boost::hash<std::pair<unsigned int, unsigned int>>>
            delta_correction_local_acc;
      } copy_data;

      auto worker = [&](decltype(combined_agglomerate_range.begin())
                            const &agglomerate_it,
                        ScratchData &, CopyData &local_copy_data) {
        local_copy_data.delta_correction_local_acc.clear();

        const auto &interior_agglomerate = agglomerate_it->get<0>();
        const auto &halo_agglomerate = agglomerate_it->get<1>();

        dealii::Triangulation<dim> interior_agglomerate_triangulation;
        std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
                 typename dealii::DoFHandler<dim>::active_cell_iterator>
            interior_patch_to_global_map;
        amge.build_agglomerate_triangulation(interior_agglomerate,
                                             interior_agglomerate_triangulation,
                                             interior_patch_to_global_map);
        dealii::Triangulation<dim> halo_agglomerate_triangulation;
        std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
                 typename dealii::DoFHandler<dim>::active_cell_iterator>
            halo_patch_to_global_map;
        amge.build_agglomerate_triangulation(halo_agglomerate,
                                             halo_agglomerate_triangulation,
                                             halo_patch_to_global_map);
        ASSERT(interior_patch_to_global.empty() ==
               halo_patch_to_global_map.empty());

        // Now that we have the triangulation, we can do the evaluation on
        // the agglomerates
        dealii::DoFHandler<dim> interior_agglomerate_dof_handler(
            interior_agglomerate_triangulation);
        interior_agglomerate_dof_handler.distribute_dofs(
            dealii_mesh_evaluator->get_dof_handler().get_fe());

        dealii::DoFHandler<dim> halo_agglomerate_dof_handler(
            halo_agglomerate_triangulation);
        halo_agglomerate_dof_handler.distribute_dofs(
            dealii_mesh_evaluator->get_dof_handler().get_fe());

        // Put the result in the matrix
        // Compute the map between the local and the global dof indices.
        std::vector<dealii::types::global_dof_index> interior_dof_indices_map =
            amge.compute_dof_index_map(interior_patch_to_global_map,
                                       interior_agglomerate_dof_handler);
        unsigned int const n_interior_elem = interior_dof_indices_map.size();

        std::vector<dealii::types::global_dof_index> halo_dof_indices_map =
            amge.compute_dof_index_map(halo_patch_to_global_map,
                                       halo_agglomerate_dof_handler);
        unsigned int const n_halo_elem = halo_dof_indices_map.size();

        unsigned int const i =
            agglomerate_it - combined_agglomerate_range.begin();
        for (unsigned int j = 0; j < n_local_eigenvectors; ++j)
        {
          unsigned int const row = i * n_local_eigenvectors + j;
          // Get the vector used for the matrix-vector multiplication
          dealii::Vector<ScalarType> interior_delta_eig(n_interior_elem);
          for (unsigned int k = 0; k < n_interior_elem; ++k)
          {
            interior_delta_eig[k] =
                delta_eigenvector_matrix->el(row, interior_dof_indices_map[k]);
          }
          dealii::Vector<ScalarType> halo_delta_eig(n_halo_elem);
          for (unsigned int k = 0; k < n_halo_elem; ++k)
          {
            if (std::find(interior_dof_indices_map.begin(),
                          interior_dof_indices_map.end(),
                          halo_dof_indices_map[k]) !=
                interior_dof_indices_map.end())
            {
              halo_delta_eig[k] =
                  delta_eigenvector_matrix->el(row, halo_dof_indices_map[k]) +
                  eigenvector_matrix->el(row, halo_dof_indices_map[k]);
            }
          }

          // Perform the matrix-vector multiplication
          dealii::Vector<ScalarType> interior_correction(n_interior_elem);
          dealii_mesh_evaluator->matrix_free_evaluate_agglomerate(
              interior_agglomerate_dof_handler, interior_delta_eig,
              interior_correction);
          dealii::Vector<ScalarType> halo_correction(n_halo_elem);
          dealii_mesh_evaluator->matrix_free_evaluate_agglomerate(
              halo_agglomerate_dof_handler, halo_delta_eig, halo_correction);

          // We would like to fill the delta correction matrix but we can't
          // because we don't know the sparsity pattern. So we accumulate
          // all the values and then fill the matrix using the set()
          // function.
          for (unsigned int k = 0; k < n_interior_elem; ++k)
          {
            local_copy_data.delta_correction_local_acc[std::make_pair(
                row, interior_dof_indices_map[k])] += interior_correction[k];
          }
          for (unsigned int k = 0; k < n_halo_elem; ++k)
          {
            local_copy_data.delta_correction_local_acc[std::make_pair(
                row, halo_dof_indices_map[k])] += halo_correction[k];
          }
        }
      };

      auto copier = [&](CopyData const &local_copy_data) {
        for (auto const &local_pair :
             local_copy_data.delta_correction_local_acc)
        {
          delta_correction_acc[local_pair.first] += local_pair.second;
        }
      };

      dealii::WorkStream::run(combined_agglomerate_range.begin(),
                              combined_agglomerate_range.end(), worker, copier,
                              scratch_data, copy_data);
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

    for (unsigned int row = 0; row < eigenvector_matrix->m(); ++row)
    {
      for (auto column_iterator = eigenvector_matrix->begin(row);
           column_iterator != eigenvector_matrix->end(row); ++column_iterator)
      {
        column_iterator->value() *= eigenvalues[row];
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
