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

#ifndef AMGE_HOST_TEMPLATES_HPP
#define AMGE_HOST_TEMPLATES_HPP

#include <mfmg/amge_host.hpp>
#include <mfmg/dealii_adapters.hpp>

#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/arpack_solver.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <EpetraExt_MatrixMatrix.h>

namespace mfmg
{
template <int dim, typename MeshEvaluator, typename VectorType>
AMGe_host<dim, MeshEvaluator, VectorType>::AMGe_host(
    MPI_Comm comm, dealii::DoFHandler<dim> const &dof_handler,
    std::string const eigensolver_type)
    : AMGe<dim, VectorType>(comm, dof_handler),
      _eigensolver_type(eigensolver_type)
{
}

template <int dim, typename MeshEvaluator, typename VectorType>
std::tuple<std::vector<std::complex<double>>,
           std::vector<dealii::Vector<double>>,
           std::vector<typename VectorType::value_type>,
           std::vector<dealii::types::global_dof_index>>
AMGe_host<dim, MeshEvaluator, VectorType>::compute_local_eigenvectors(
    unsigned int n_eigenvectors, double tolerance,
    dealii::Triangulation<dim> const &agglomerate_triangulation,
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator> const
        &patch_to_global_map,
    const MeshEvaluator &evaluator) const
{
  dealii::DoFHandler<dim> agglomerate_dof_handler(agglomerate_triangulation);
  dealii::ConstraintMatrix agglomerate_constraints;

  DealIIMesh<dim> agglomerate_mesh(agglomerate_dof_handler,
                                   agglomerate_constraints);

  // Call user function to fill in the matrix and build the mass matrix
  auto agglomerate_operator = evaluator.get_local_operator(agglomerate_mesh);

  auto agglomerate_system_matrix = agglomerate_operator->get_matrix();

  // Get the diagonal elements
  unsigned int const size = agglomerate_system_matrix->m();
  std::vector<ScalarType> diag_elements(size);
  for (unsigned int i = 0; i < size; ++i)
    diag_elements[i] = agglomerate_system_matrix->diag_element(i);

  // Compute the eigenvalues and the eigenvectors
  unsigned int const n_dofs_agglomerate = agglomerate_system_matrix->m();
  std::vector<std::complex<double>> eigenvalues(n_eigenvectors);
  // Arpack only works with double not float
  std::vector<dealii::Vector<double>> eigenvectors(
      n_eigenvectors, dealii::Vector<double>(n_dofs_agglomerate));

  if (_eigensolver_type == "arpack")
  {
    // Make Identity mass matrix
    dealii::SparsityPattern agglomerate_mass_sparsity_pattern;
    dealii::SparseMatrix<ScalarType> agglomerate_mass_matrix;
    std::vector<std::vector<unsigned int>> column_indices(
        size, std::vector<unsigned int>(1));
    for (unsigned int i = 0; i < size; ++i)
      column_indices[i][0] = i;
    agglomerate_mass_sparsity_pattern.copy_from(
        size, size, column_indices.begin(), column_indices.end());
    agglomerate_mass_matrix.reinit(agglomerate_mass_sparsity_pattern);
    for (unsigned int i = 0; i < size; ++i)
      agglomerate_mass_matrix.diag_element(i) = 1.;
    dealii::SparseDirectUMFPACK inv_system_matrix;
    inv_system_matrix.initialize(*agglomerate_system_matrix);

    dealii::SolverControl solver_control(n_dofs_agglomerate, tolerance);
    unsigned int const n_arnoldi_vectors = 2 * n_eigenvectors + 2;
    bool const symmetric = true;
    // We want the eigenvalues of the smallest magnitudes but we need to ask for
    // the ones with the largest magnitudes because they are computed for the
    // inverse of the matrix we care about.
    dealii::ArpackSolver::WhichEigenvalues which_eigenvalues =
        dealii::ArpackSolver::WhichEigenvalues::largest_magnitude;
    dealii::ArpackSolver::AdditionalData additional_data(
        n_arnoldi_vectors, which_eigenvalues, symmetric);
    dealii::ArpackSolver solver(solver_control, additional_data);

    // Compute the eigenvectors. Arpack outputs eigenvectors with a L2 norm of
    // one.
    dealii::Vector<double> initial_vector(n_dofs_agglomerate);
    evaluator.set_initial_guess(agglomerate_mesh, initial_vector);
    solver.set_initial_vector(initial_vector);
    solver.solve(*agglomerate_system_matrix, agglomerate_mass_matrix,
                 inv_system_matrix, eigenvalues, eigenvectors);
  }
  else
  {
    // Use Lapack to compute the eigenvalues
    dealii::LAPACKFullMatrix<double> full_matrix;
    full_matrix.copy_from(*agglomerate_system_matrix);

    double const lower_bound = -0.5;
    double const upper_bound = 100.;
    double const tol = 1e-12;
    dealii::Vector<double> lapack_eigenvalues(size);
    dealii::FullMatrix<double> lapack_eigenvectors;
    full_matrix.compute_eigenvalues_symmetric(
        lower_bound, upper_bound, tol, lapack_eigenvalues, lapack_eigenvectors);

    // Copy the eigenvalues and the eigenvectors in the right format
    for (unsigned int i = 0; i < n_eigenvectors; ++i)
      eigenvalues[i] = lapack_eigenvalues[i];

    for (unsigned int i = 0; i < n_eigenvectors; ++i)
      for (unsigned int j = 0; j < n_dofs_agglomerate; ++j)
        eigenvectors[i][j] = lapack_eigenvectors[j][i];
  }

  // Compute the map between the local and the global dof indices.
  std::vector<dealii::types::global_dof_index> dof_indices_map =
      this->compute_dof_index_map(patch_to_global_map, agglomerate_dof_handler);

  return std::make_tuple(eigenvalues, eigenvectors, diag_elements,
                         dof_indices_map);
}

template <int dim, typename MeshEvaluator, typename VectorType>
void AMGe_host<dim, MeshEvaluator, VectorType>::
    compute_restriction_sparse_matrix(
        std::vector<dealii::Vector<double>> const &eigenvectors,
        std::vector<std::vector<typename VectorType::value_type>> const
            &diag_elements,
        std::vector<std::vector<dealii::types::global_dof_index>> const
            &dof_indices_maps,
        std::vector<unsigned int> const &n_local_eigenvectors,
        dealii::TrilinosWrappers::SparseMatrix const &system_sparse_matrix,
        dealii::TrilinosWrappers::SparseMatrix &restriction_sparse_matrix) const
{
  // Extract the diagonal of the system sparse matrix. Each processor gets the
  // locally relevant indices, i.e., owned + ghost
  dealii::IndexSet locally_owned_dofs =
      system_sparse_matrix.locally_owned_domain_indices();
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(this->_dof_handler,
                                                  locally_relevant_dofs);
  VectorType locally_owned_global_diag(locally_owned_dofs, this->_comm);
  for (auto const val : locally_owned_dofs)
    locally_owned_global_diag[val] = system_sparse_matrix.diag_element(val);
  locally_owned_global_diag.compress(dealii::VectorOperation::insert);

  VectorType locally_relevant_global_diag(locally_owned_dofs,
                                          locally_relevant_dofs, this->_comm);
  locally_relevant_global_diag = locally_owned_global_diag;

  // Compute the sparsity pattern (Epetra_FECrsGraph)
  dealii::TrilinosWrappers::SparsityPattern restriction_sp =
      compute_restriction_sparsity_pattern(eigenvectors, dof_indices_maps,
                                           n_local_eigenvectors);

  // Build the restriction sparse matrix
  restriction_sparse_matrix.reinit(restriction_sp);
  unsigned int const n_local_rows(eigenvectors.size());
  std::pair<dealii::types::global_dof_index,
            dealii::types::global_dof_index> const local_range =
      restriction_sp.local_range();
  unsigned int const n_agglomerates = n_local_eigenvectors.size();
  unsigned int pos = 0;
  for (unsigned int i = 0; i < n_agglomerates; ++i)
  {
    unsigned int const n_local_eig = n_local_eigenvectors[i];
    for (unsigned int k = 0; k < n_local_eig; ++k)
    {
      unsigned int const n_elem = eigenvectors[pos].size();
      for (unsigned int j = 0; j < n_elem; ++j)
      {
        dealii::types::global_dof_index const global_pos =
            dof_indices_maps[i][j];
        restriction_sparse_matrix.add(
            local_range.first + pos, global_pos,
            diag_elements[i][j] / locally_relevant_global_diag[global_pos] *
                eigenvectors[pos][j]);
      }
      ++pos;
    }
  }

  // Compress the matrix
  restriction_sparse_matrix.compress(dealii::VectorOperation::add);

#if MFMG_DEBUG
  // Check that the sum of the weight matrices is the identity
  dealii::TrilinosWrappers::SparsityPattern sp(locally_owned_dofs,
                                               locally_owned_dofs, this->_comm);
  for (auto local_index : locally_owned_dofs)
    sp.add(local_index, local_index);
  sp.compress();

  dealii::TrilinosWrappers::SparseMatrix weight_matrix(sp);
  pos = 0;
  for (unsigned int i = 0; i < n_agglomerates; ++i)
  {
    unsigned int const n_local_eig = n_local_eigenvectors[i];
    unsigned int const n_elem = eigenvectors[pos].size();
    for (unsigned int j = 0; j < n_elem; ++j)
    {
      dealii::types::global_dof_index const global_pos = dof_indices_maps[i][j];
      double const value =
          diag_elements[i][j] / locally_relevant_global_diag[global_pos];
      weight_matrix.add(global_pos, global_pos, value);
    }
    ++pos;
  }

  // Compress the matrix
  weight_matrix.compress(dealii::VectorOperation::add);

  for (auto index : locally_owned_dofs)
    ASSERT(std::abs(weight_matrix.diag_element(index) - 1.0) < 1e-14,
           "Sum of local weight matrices is not the identity");
#endif
}

template <int dim, typename MeshEvaluator, typename VectorType>
void AMGe_host<dim, MeshEvaluator, VectorType>::setup_restrictor(
    std::array<unsigned int, dim> const &agglomerate_dim,
    unsigned int const n_eigenvectors, double const tolerance,
    MeshEvaluator const &evaluator,
    std::shared_ptr<typename MeshEvaluator::global_operator_type const>
        global_operator,
    dealii::TrilinosWrappers::SparseMatrix &restriction_sparse_matrix)
{
  // Flag the cells to build agglomerates.
  unsigned int const n_agglomerates = this->build_agglomerates(agglomerate_dim);

  // Parallel part of the setup.
  std::vector<unsigned int> agglomerate_ids(n_agglomerates);
  std::iota(agglomerate_ids.begin(), agglomerate_ids.end(), 1);
  std::vector<dealii::Vector<double>> eigenvectors;
  std::vector<std::vector<ScalarType>> diag_elements;
  std::vector<std::vector<dealii::types::global_dof_index>> dof_indices_maps;
  std::vector<unsigned int> n_local_eigenvectors;
  CopyData copy_data;
  dealii::WorkStream::run(
      agglomerate_ids.begin(), agglomerate_ids.end(),
      static_cast<
          std::function<void(std::vector<unsigned int>::iterator const &,
                             ScratchData &, CopyData &)>>(
          std::bind(&AMGe_host::local_worker, *this, n_eigenvectors, tolerance,
                    std::cref(evaluator), std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3)),
      static_cast<std::function<void(CopyData const &)>>(std::bind(
          &AMGe_host::copy_local_to_global, *this, std::placeholders::_1,
          std::ref(eigenvectors), std::ref(diag_elements),
          std::ref(dof_indices_maps), std::ref(n_local_eigenvectors))),
      ScratchData(), copy_data);

  // Return to a serial execution
  auto system_sparse_matrix = global_operator->get_matrix();
  compute_restriction_sparse_matrix(
      eigenvectors, diag_elements, dof_indices_maps, n_local_eigenvectors,
      *system_sparse_matrix, restriction_sparse_matrix);
}

template <int dim, typename MeshEvaluator, typename VectorType>
void AMGe_host<dim, MeshEvaluator, VectorType>::local_worker(
    unsigned int const n_eigenvectors, double const tolerance,
    MeshEvaluator const &evaluator,
    std::vector<unsigned int>::iterator const &agg_id, ScratchData &,
    CopyData &copy_data)
{
  dealii::Triangulation<dim> agglomerate_triangulation;
  std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
           typename dealii::DoFHandler<dim>::active_cell_iterator>
      agglomerate_to_global_tria_map;

  this->build_agglomerate_triangulation(*agg_id, agglomerate_triangulation,
                                        agglomerate_to_global_tria_map);

  // We ignore the eigenvalues.
  std::tie(std::ignore, copy_data.local_eigenvectors, copy_data.diag_elements,
           copy_data.local_dof_indices_map) =
      compute_local_eigenvectors(n_eigenvectors, tolerance,
                                 agglomerate_triangulation,
                                 agglomerate_to_global_tria_map, evaluator);
}

template <int dim, typename MeshEvaluator, typename VectorType>
void AMGe_host<dim, MeshEvaluator, VectorType>::copy_local_to_global(
    CopyData const &copy_data,
    std::vector<dealii::Vector<double>> &eigenvectors,
    std::vector<std::vector<typename VectorType::value_type>> &diag_elements,
    std::vector<std::vector<dealii::types::global_dof_index>> &dof_indices_maps,
    std::vector<unsigned int> &n_local_eigenvectors)
{
  eigenvectors.insert(eigenvectors.end(), copy_data.local_eigenvectors.begin(),
                      copy_data.local_eigenvectors.end());

  diag_elements.push_back(copy_data.diag_elements);

  dof_indices_maps.push_back(copy_data.local_dof_indices_map);

  n_local_eigenvectors.push_back(copy_data.local_eigenvectors.size());
}

template <int dim, typename MeshEvaluator, typename VectorType>
dealii::TrilinosWrappers::SparsityPattern
AMGe_host<dim, MeshEvaluator, VectorType>::compute_restriction_sparsity_pattern(
    std::vector<dealii::Vector<double>> const &eigenvectors,
    std::vector<std::vector<dealii::types::global_dof_index>> const
        &dof_indices_maps,
    std::vector<unsigned int> const &n_local_eigenvectors) const
{
  // Compute the row IndexSet
  int const n_procs = dealii::Utilities::MPI::n_mpi_processes(this->_comm);
  int const rank = dealii::Utilities::MPI::this_mpi_process(this->_comm);
  unsigned int const n_local_rows(eigenvectors.size());
  std::vector<unsigned int> n_rows_per_proc(n_procs);
  n_rows_per_proc[rank] = n_local_rows;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &n_rows_per_proc[0], 1,
                MPI_UNSIGNED, this->_comm);

  dealii::types::global_dof_index n_total_rows =
      std::accumulate(n_rows_per_proc.begin(), n_rows_per_proc.end(),
                      static_cast<dealii::types::global_dof_index>(0));
  dealii::types::global_dof_index n_rows_before =
      std::accumulate(n_rows_per_proc.begin(), n_rows_per_proc.begin() + rank,
                      static_cast<dealii::types::global_dof_index>(0));
  dealii::IndexSet row_indexset(n_total_rows);
  row_indexset.add_range(n_rows_before, n_rows_before + n_local_rows);
  row_indexset.compress();

  // Build the sparsity pattern
  dealii::TrilinosWrappers::SparsityPattern sp(
      row_indexset, this->_dof_handler.locally_owned_dofs(), this->_comm);

  unsigned int const n_agglomerates = n_local_eigenvectors.size();
  unsigned int row = 0;
  for (unsigned int i = 0; i < n_agglomerates; ++i)
  {
    unsigned int const n_local_eig = n_local_eigenvectors[i];
    for (unsigned int j = 0; j < n_local_eig; ++j)
    {
      sp.add_entries(n_rows_before + row, dof_indices_maps[i].begin(),
                     dof_indices_maps[i].end());
      ++row;
    }
  }

  sp.compress();

  return sp;
}
}

#endif
