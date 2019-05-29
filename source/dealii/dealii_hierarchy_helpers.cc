/**************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                            *
 * All rights reserved.                                                   *
 *                                                                        *
 * This file is part of the mfmg library. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the  *
 * top-level directory                                                    *
 *                                                                        *
 * SPDX-License-Identifier: BSD-3-Clause                                  *
 *************************************************************************/

#include <mfmg/common/instantiation.hpp>
#include <mfmg/common/operator.hpp>
#include <mfmg/dealii/dealii_hierarchy_helpers.hpp>
#include <mfmg/dealii/dealii_smoother.hpp>
#include <mfmg/dealii/dealii_solver.hpp>
#include <mfmg/dealii/dealii_trilinos_matrix_operator.hpp>
#include <mfmg/dealii/dealii_utils.hpp>

#include <deal.II/base/work_stream.h>
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
  // Downcast to DealIIMeshEvaluator
  auto dealii_mesh_evaluator =
      std::dynamic_pointer_cast<DealIIMeshEvaluator<dim>>(mesh_evaluator);

  auto system_matrix =
      std::make_shared<dealii::TrilinosWrappers::SparseMatrix>();

  // Call user function to fill in the system matrix
  dealii_mesh_evaluator->evaluate_global(
      dealii_mesh_evaluator->get_dof_handler(),
      dealii_mesh_evaluator->get_constraints(), *system_matrix);

  std::shared_ptr<Operator<VectorType>> global_operator =
      std::make_shared<DealIITrilinosMatrixOperator<VectorType>>(system_matrix);

  return global_operator;
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
  auto agglomerate_params = params->get_child("agglomeration");
  if (fast_ap)
  {
    AMGe_host<dim, DealIIMeshEvaluator<dim>, VectorType> amge(
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
        std::vector<dealii::types::global_dof_index> rows;
        std::vector<dealii::types::global_dof_index> cols;
        std::vector<std::vector<dealii::TrilinosScalar>> values_per_row;
      } copy_data;

      auto worker =
          [&](const std::vector<std::vector<unsigned int>>::const_iterator
                  &agglomerate_it,
              ScratchData &, CopyData &local_copy_data) {
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
            dealii::AffineConstraints<double> agglomerate_constraints;
            dealii::SparsityPattern agglomerate_sparsity_pattern;
            dealii::SparseMatrix<ScalarType> agglomerate_system_matrix;
            // Call user function to build the system matrix
            dealii_mesh_evaluator->evaluate_agglomerate(
                agglomerate_dof_handler, agglomerate_constraints,
                agglomerate_sparsity_pattern, agglomerate_system_matrix);

            // Put the result in the matrix
            // Compute the map between the local and the global dof indices.
            local_copy_data.rows.resize(n_local_eigenvectors);

            local_copy_data.cols = amge.compute_dof_index_map(
                patch_to_global_map, agglomerate_dof_handler);
            auto const &dof_indices_map = local_copy_data.cols;
            unsigned int const n_elem = dof_indices_map.size();

            // We need a clean reset for the values we are going to store.
            // Otherwise, we would accumulate values across patches
            // corresponding to different degrees of freedom.
            local_copy_data.values_per_row.resize(n_local_eigenvectors);
            std::fill(local_copy_data.values_per_row.begin(),
                      local_copy_data.values_per_row.end(),
                      std::vector<dealii::TrilinosScalar>(n_elem));

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
              agglomerate_system_matrix.vmult(correction, delta_eig);

              // Store the values the delta correction matrix is to be filled
              // with.
              local_copy_data.rows[j] = global_row;
              std::transform(correction.begin(), correction.end(),
                             local_copy_data.values_per_row[j].begin(),
                             local_copy_data.values_per_row[j].begin(),
                             std::plus<double>());
            }
          };

      auto copier = [&](const CopyData &local_copy_data) {
        for (unsigned int i = 0; i < local_copy_data.rows.size(); ++i)
        {
          // Since we didn't provide a sparsity pattern this "set" operation
          // actually acts as an "add" operation until "compress" is called.
          delta_correction_matrix.set(
              local_copy_data.rows[i], local_copy_data.cols.size(),
              local_copy_data.cols.data(),
              local_copy_data.values_per_row[i].data(), true);
        }
      };

      dealii::WorkStream::run(agglomerates_vector.begin(),
                              agglomerates_vector.end(), worker, copier,
                              scratch_data, copy_data);

      is_halo_agglomerate = true;
    }

    // Fill delta_correction_matrix
    delta_correction_matrix.compress(dealii::VectorOperation::insert);

    // Scale the eigenvectors by their corresponding eigenvalues.
    // It turned out that accessing the individual matrix entries through the
    // deal.II TrilinosWrapper::SparseMatrix::begin() interface is much slower
    // than constructing a representation of the eigenvalues as
    // Epetra_MultiVector and calling a Epetra_CrsMatrix member function.
    auto const range_start = eigenvector_matrix->local_range().first;
    auto const range_end = eigenvector_matrix->local_range().second;

    dealii::IndexSet locally_owned_elements(eigenvector_matrix->m());
    locally_owned_elements.add_range(range_start, range_end);
    dealii::TrilinosWrappers::MPI::Vector vector_eigenvalues(
        locally_owned_elements, comm);
    std::vector<dealii::types::global_dof_index> local_indices(
        eigenvalues.size());
    std::iota(local_indices.begin(), local_indices.end(), range_start);
    vector_eigenvalues.set(local_indices, eigenvalues);
    vector_eigenvalues.compress(dealii::VectorOperation::insert);

    Epetra_MultiVector &dummy_multi_vector =
        vector_eigenvalues.trilinos_vector();
    auto &dummy_matrix =
        const_cast<Epetra_CrsMatrix &>(eigenvector_matrix->trilinos_matrix());
    dummy_matrix.LeftScale(*dummy_multi_vector(0));

    eigenvector_matrix->compress(dealii::VectorOperation::insert);

    // Add the eigenvector matrix and the delta correction matrix to create ap
    Epetra_CrsMatrix *ap = nullptr;
    // We want to use functions that have been deprecated in deal.II but they
    // won't be removed in the foreseeable future
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    Epetra_Map range_map = eigenvector_matrix->domain_partitioner();
    Epetra_Map domain_map = eigenvector_matrix->range_partitioner();
#pragma GCC diagnostic pop
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
    AMGe_host<dim, DealIIMeshEvaluator<dim>, VectorType> amge(
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

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
DealIIHierarchyHelpers<dim, VectorType>::fast_multiply_transpose()
{
  return _ap_operator;
}

} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_DIM_VECTORTYPE(TUPLE(DealIIHierarchyHelpers))
