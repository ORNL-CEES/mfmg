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

#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/lac/arpack_solver.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_direct.h>

#include <EpetraExt_MatrixMatrix.h>

namespace mfmg
{
template <int dim, typename VectorType>
AMGe_host<dim, VectorType>::AMGe_host(
    MPI_Comm comm, dealii::DoFHandler<dim> const &dof_handler)
    : AMGe<dim, VectorType>(comm, dof_handler)
{
}

template <int dim, typename VectorType>
AMGe_host<dim, VectorType>::AMGe_host(AMGe_host<dim, VectorType> const &other)
    : AMGe<dim, VectorType>(other._comm, other._dof_handler)
{
  _system_matrix_ptr = other._system_matrix_ptr;
  _restriction_sparse_matrix.copy_from(other._restriction_sparse_matrix);
  _coarse_sparse_matrix.copy_from(other._coarse_sparse_matrix);
}

template <int dim, typename VectorType>
std::tuple<std::vector<std::complex<double>>,
           std::vector<dealii::Vector<double>>,
           std::vector<dealii::types::global_dof_index>>
AMGe_host<dim, VectorType>::compute_local_eigenvectors(
    unsigned int n_eigenvalues, double tolerance,
    dealii::Triangulation<dim> const &agglomerate_triangulation,
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator> const
        &patch_to_global_map,
    std::function<void(
        dealii::DoFHandler<dim> &, dealii::ConstraintMatrix &,
        dealii::SparsityPattern &, dealii::SparseMatrix<ScalarType> &,
        dealii::SparsityPattern &, dealii::SparseMatrix<ScalarType> &)> const
        &evaluate) const
{
  dealii::SparsityPattern system_sparsity_pattern;
  dealii::SparsityPattern mass_sparsity_pattern;
  dealii::SparseMatrix<ScalarType> agglomerate_system_matrix;
  dealii::SparseMatrix<ScalarType> agglomerate_mass_matrix;
  dealii::ConstraintMatrix agglomerate_constraints;

  dealii::DoFHandler<dim> agglomerate_dof_handler(agglomerate_triangulation);

  // Call user function to fill in the matrix and build the mass matrix
  evaluate(agglomerate_dof_handler, agglomerate_constraints,
           system_sparsity_pattern, agglomerate_system_matrix,
           mass_sparsity_pattern, agglomerate_mass_matrix);

  dealii::SparseDirectUMFPACK inv_system_matrix;
  inv_system_matrix.initialize(agglomerate_system_matrix);

  // Compute the eigenvalues and the eigenvectors
  unsigned int const n_dofs_agglomerate = agglomerate_system_matrix.m();
  std::vector<std::complex<double>> eigenvalues(n_eigenvalues);
  // Arpack only works with double not float
  std::vector<dealii::Vector<double>> eigenvectors(
      n_eigenvalues, dealii::Vector<double>(n_dofs_agglomerate));

  dealii::SolverControl solver_control(n_dofs_agglomerate, tolerance);
  unsigned int const n_arnoldi_vectors = 2 * n_eigenvalues + 2;
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
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0., 1.);
  dealii::Vector<double> initial_vector(n_dofs_agglomerate);
  for (unsigned int i = 0; i < n_dofs_agglomerate; ++i)
    if (agglomerate_constraints.is_constrained(i) == false)
      initial_vector[i] = distribution(generator);
  solver.set_initial_vector(initial_vector);
  solver.solve(agglomerate_system_matrix, agglomerate_mass_matrix,
               inv_system_matrix, eigenvalues, eigenvectors);

  // Compute the map between the local and the global dof indices.
  std::vector<dealii::types::global_dof_index> dof_indices_map =
      this->compute_dof_index_map(patch_to_global_map, agglomerate_dof_handler);

  return std::make_tuple(eigenvalues, eigenvectors, dof_indices_map);
}

template <int dim, typename VectorType>
void AMGe_host<dim, VectorType>::compute_restriction_sparse_matrix(
    std::vector<dealii::Vector<double>> const &eigenvectors,
    std::vector<std::vector<dealii::types::global_dof_index>> const
        &dof_indices_maps,
    dealii::TrilinosWrappers::SparseMatrix &restriction_sparse_matrix) const
{
  // Compute the sparsity pattern (Epetra_FECrsGraph)
  dealii::TrilinosWrappers::SparsityPattern restriction_sp =
      compute_restriction_sparsity_pattern(eigenvectors, dof_indices_maps);

  // Build the restriction sparse matrix
  restriction_sparse_matrix.reinit(restriction_sp);
  unsigned int const n_local_rows(eigenvectors.size());
  std::pair<dealii::types::global_dof_index,
            dealii::types::global_dof_index> const local_range =
      restriction_sp.local_range();

  for (unsigned int i = 0; i < n_local_rows; ++i)
  {
    restriction_sparse_matrix.set(
        local_range.first + i, dof_indices_maps[i].size(),
        &(dof_indices_maps[i][0]), eigenvectors[i].begin());
  }

  // Compress the matrix
  restriction_sparse_matrix.compress(dealii::VectorOperation::insert);
}

template <int dim, typename VectorType>
void AMGe_host<dim, VectorType>::setup(
    std::array<unsigned int, dim> const &agglomerate_dim,
    unsigned int const n_eigenvalues, double const tolerance,
    std::function<void(dealii::DoFHandler<dim> &dof_handler,
                       dealii::ConstraintMatrix &,
                       dealii::SparsityPattern &system_sparsity_pattern,
                       dealii::SparseMatrix<ScalarType> &,
                       dealii::SparsityPattern &mass_sparsity_pattern,
                       dealii::SparseMatrix<ScalarType> &)> const &evaluate,
    dealii::TrilinosWrappers::SparseMatrix const &system_sparse_matrix)
{
  // Flag the cells to build agglomerates.
  unsigned int const n_agglomerates = this->build_agglomerates(agglomerate_dim);

  // Parallel part of the setup.
  std::vector<unsigned int> agglomerate_ids(n_agglomerates);
  std::iota(agglomerate_ids.begin(), agglomerate_ids.end(), 1);
  std::vector<dealii::Vector<double>> eigenvectors;
  std::vector<std::vector<dealii::types::global_dof_index>> dof_indices_maps;
  CopyData copy_data;
  dealii::WorkStream::run(
      agglomerate_ids.begin(), agglomerate_ids.end(),
      static_cast<
          std::function<void(std::vector<unsigned int>::iterator const &,
                             ScratchData &, CopyData &)>>(
          std::bind(&AMGe_host::local_worker, *this, n_eigenvalues, tolerance,
                    std::cref(evaluate), std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3)),
      static_cast<std::function<void(CopyData const &)>>(std::bind(
          &AMGe_host::copy_local_to_global, *this, std::placeholders::_1,
          std::ref(eigenvectors), std::ref(dof_indices_maps))),
      ScratchData(), copy_data);

  // Return to a serial execution
  compute_restriction_sparse_matrix(eigenvectors, dof_indices_maps,
                                    _restriction_sparse_matrix);

  // Build the coarse sparse matrix, i.e, RAP = RAR^T
  // Compute AR^T
  _system_matrix_ptr = &system_sparse_matrix;
  dealii::TrilinosWrappers::SparseMatrix tmp_sparse_matrix(
      _system_matrix_ptr->locally_owned_range_indices(),
      _restriction_sparse_matrix.locally_owned_range_indices(),
      _system_matrix_ptr->get_mpi_communicator());
  EpetraExt::MatrixMatrix::Multiply(
      _system_matrix_ptr->trilinos_matrix(), false,
      _restriction_sparse_matrix.trilinos_matrix(), true,
      const_cast<Epetra_CrsMatrix &>(tmp_sparse_matrix.trilinos_matrix()));

  // Compute R(AR^T)
  _restriction_sparse_matrix.mmult(_coarse_sparse_matrix, tmp_sparse_matrix);
}

template <int dim, typename VectorType>
void AMGe_host<dim, VectorType>::local_worker(
    unsigned int const n_eigenvalues, double const tolerance,
    std::function<void(dealii::DoFHandler<dim> &dof_handler,
                       dealii::ConstraintMatrix &,
                       dealii::SparsityPattern &system_sparsity_pattern,
                       dealii::SparseMatrix<ScalarType> &,
                       dealii::SparsityPattern &mass_sparsity_pattern,
                       dealii::SparseMatrix<ScalarType> &)> const &evaluate,
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
  std::tie(std::ignore, copy_data.local_eigenvectors,
           copy_data.local_dof_indices_map) =
      compute_local_eigenvectors(n_eigenvalues, tolerance,
                                 agglomerate_triangulation,
                                 agglomerate_to_global_tria_map, evaluate);
}

template <int dim, typename VectorType>
void AMGe_host<dim, VectorType>::copy_local_to_global(
    CopyData const &copy_data,
    std::vector<dealii::Vector<double>> &eigenvectors,
    std::vector<std::vector<dealii::types::global_dof_index>> &dof_indices_maps)
{
  eigenvectors.insert(eigenvectors.end(), copy_data.local_eigenvectors.begin(),
                      copy_data.local_eigenvectors.end());
  unsigned int const n_local_eigenvectors = copy_data.local_eigenvectors.size();
  for (unsigned int i = 0; i < n_local_eigenvectors; ++i)
    dof_indices_maps.push_back(copy_data.local_dof_indices_map);
}

template <int dim, typename VectorType>
dealii::TrilinosWrappers::SparsityPattern
AMGe_host<dim, VectorType>::compute_restriction_sparsity_pattern(
    std::vector<dealii::Vector<double>> const &eigenvectors,
    std::vector<std::vector<dealii::types::global_dof_index>> const
        &dof_indices_maps) const
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

  for (unsigned int i = 0; i < n_local_rows; ++i)
    sp.add_entries(n_rows_before + i, dof_indices_maps[i].begin(),
                   dof_indices_maps[i].end());

  sp.compress();

  return sp;
}
}

#endif
