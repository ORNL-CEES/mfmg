/*************************************************************************
 * Copyright (c) 2017 by the mfmg authors                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#ifndef AMG_TEMPLATES_HPP
#define AMG_TEMPLATES_HPP

#include <mfmg/amge.hpp>

#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/arpack_solver.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out.h>

#include <algorithm>
#include <fstream>

namespace mfmg
{
template <int dim, typename ScalarType>
AMGe<dim, ScalarType>::AMGe(MPI_Comm comm,
                            dealii::DoFHandler<dim> const &dof_handler)
    : _comm(comm), _dof_handler(dof_handler)
{
}

template <int dim, typename ScalarType>
unsigned int AMGe<dim, ScalarType>::build_agglomerates(
    std::array<unsigned int, dim> const &agglomerate_dim) const
{
  // Faces in deal.II are orderd as follows: left (x_m) = 0, right (x_p) = 1,
  // front (y_m) = 2, back (y_p) = 3, bottom (z_m) = 4, top (z_p) = 5
  unsigned int constexpr x_p = 1;
  unsigned int constexpr y_p = 3;
  unsigned int constexpr z_p = 5;

  // Flag the cells to create the agglomerates
  unsigned int agglomerate = 1;
  for (auto cell : _dof_handler.active_cell_iterators())
  {
    if ((cell->is_locally_owned()) && (cell->user_index() == 0))
    {
#if MFMG_DEBUG
      int const cell_level = cell->level();
#endif
      cell->set_user_index(agglomerate);
      auto current_z_cell = cell;
      unsigned int const d_3 = (dim < 3) ? 1 : agglomerate_dim.back();
      for (unsigned int k = 0; k < d_3; ++k)
      {
        auto current_y_cell = current_z_cell;
        for (unsigned int j = 0; j < agglomerate_dim[1]; ++j)
        {
          auto current_cell = current_y_cell;
          for (unsigned int i = 0; i < agglomerate_dim[0]; ++i)
          {
            current_cell->set_user_index(agglomerate);
            if (current_cell->at_boundary(x_p) == false)
            {
              // TODO For now, we assume that there is no adaptive refinement.
              // When we change this, we will need to switch to hp::DoFHandler
              auto neighbor_cell = current_cell->neighbor(x_p);
#if MFMG_DEBUG
              if ((!neighbor_cell->active()) ||
                  (neighbor_cell->level() != cell_level))
                throw std::runtime_error("Mesh locally refined");
#endif
              if (neighbor_cell->is_locally_owned())
                current_cell = neighbor_cell;
            }
            else
              break;
          }
          if (current_y_cell->at_boundary(y_p) == false)
          {
            auto neighbor_y_cell = current_y_cell->neighbor(y_p);
            if (neighbor_y_cell->is_locally_owned())
            {
#if MFMG_DEBUG
              if ((!neighbor_y_cell->active()) ||
                  (neighbor_y_cell->level() != cell_level))
                throw std::runtime_error("Mesh locally refined");
#endif
              current_y_cell = neighbor_y_cell;
            }
          }
          else
            break;
        }
        if ((dim == 3) && (current_z_cell->at_boundary(z_p) == false))
        {
          auto neighbor_z_cell = current_z_cell->neighbor(z_p);
          if (neighbor_z_cell->is_locally_owned())
          {
#if MFMG_DEBUG
            if ((!neighbor_z_cell->active()) ||
                (neighbor_z_cell->level() != cell_level))
              throw std::runtime_error("Mesh locally refined");
#endif
            current_z_cell = neighbor_z_cell;
          }
        }
        else
          break;
      }

      ++agglomerate;
    }
  }

  return agglomerate - 1;
}

template <int dim, typename ScalarType>
std::tuple<dealii::Triangulation<dim>,
           std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
                    typename dealii::DoFHandler<dim>::active_cell_iterator>>
AMGe<dim, ScalarType>::build_agglomerate_triangulation(
    unsigned int agglomerate_id) const
{
  std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>
      agglomerate;
  for (auto cell : _dof_handler.active_cell_iterators())
    if (cell->user_index() == agglomerate_id)
      agglomerate.push_back(cell);

  dealii::Triangulation<dim> agglomerate_triangulation;
  std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
           typename dealii::DoFHandler<dim>::active_cell_iterator>
      agglomerate_to_global_tria_map;

  // If the agglomerate has hanging nodes, the patch is bigger than
  // what we may expect because we cannot a create a coarse triangulation with
  // hanging nodes. Thus, we need to use FE_Nothing to get ride of unwanted
  // cells.
  dealii::GridTools::build_triangulation_from_patch<dealii::DoFHandler<dim>>(
      agglomerate, agglomerate_triangulation, agglomerate_to_global_tria_map);

  // The std::move inhibits copy elision but the code does not work otherwise
  return std::make_tuple(std::move(agglomerate_triangulation),
                         agglomerate_to_global_tria_map);
}

template <int dim, typename ScalarType>
std::tuple<std::vector<std::complex<double>>,
           std::vector<dealii::Vector<double>>,
           std::vector<dealii::types::global_dof_index>>
AMGe<dim, ScalarType>::compute_local_eigenvectors(
    unsigned int n_eigenvalues, double tolerance,
    dealii::Triangulation<dim> const &agglomerate_triangulation,
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator> const
        &patch_to_global_map,
    std::function<void(dealii::DoFHandler<dim> &, dealii::SparsityPattern &,
                       dealii::SparseMatrix<ScalarType> &,
                       dealii::SparsityPattern &,
                       dealii::SparseMatrix<ScalarType> &,
                       dealii::ConstraintMatrix &)> const &evaluate) const
{
  dealii::SparsityPattern system_sparsity_pattern;
  dealii::SparsityPattern mass_sparsity_pattern;
  dealii::SparseMatrix<ScalarType> agglomerate_system_matrix;
  dealii::SparseMatrix<ScalarType> agglomerate_mass_matrix;
  dealii::ConstraintMatrix agglomerate_constraints;

  dealii::DoFHandler<dim> agglomerate_dof_handler(agglomerate_triangulation);

  // Call user function to fill in the matrix and build the mass matrix
  evaluate(agglomerate_dof_handler, system_sparsity_pattern,
           agglomerate_system_matrix, mass_sparsity_pattern,
           agglomerate_mass_matrix, agglomerate_constraints);

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
      compute_dof_index_map(patch_to_global_map, agglomerate_dof_handler);

  return std::make_tuple(eigenvalues, eigenvectors, dof_indices_map);
}

template <int dim, typename ScalarType>
void AMGe<dim, ScalarType>::compute_restriction_sparse_matrix(
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

template <int dim, typename ScalarType>
void AMGe<dim, ScalarType>::output(std::string const &filename) const
{
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(_dof_handler);

  unsigned int const n_active_cells =
      _dof_handler.get_triangulation().n_active_cells();
  dealii::Vector<float> subdomain(n_active_cells);
  for (unsigned int i = 0; i < n_active_cells; ++i)
    subdomain(i) = _dof_handler.get_triangulation().locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  dealii::Vector<float> agglomerates(n_active_cells);
  unsigned int n = 0;
  for (auto cell : _dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
      agglomerates(n) = cell->user_index();
    ++n;
  }
  data_out.add_data_vector(agglomerates, "agglomerates");

  data_out.build_patches();

  std::string full_filename =
      filename +
      std::to_string(
          _dof_handler.get_triangulation().locally_owned_subdomain());
  std::ofstream output((full_filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (dealii::Utilities::MPI::this_mpi_process(_comm) == 0)
  {
    unsigned int const comm_size =
        dealii::Utilities::MPI::n_mpi_processes(_comm);
    std::vector<std::string> full_filenames(comm_size);
    for (unsigned int i = 0; i < comm_size; ++i)
      full_filenames[0] = filename + std::to_string(i) + ".vtu";
    std::ofstream master_output(filename + ".pvtu");
    data_out.write_pvtu_record(master_output, full_filenames);
  }
}

template <int dim, typename ScalarType>
void AMGe<dim, ScalarType>::setup(
    std::array<unsigned int, dim> const &agglomerate_dim)
{
  // Flag the cells to build agglomerates.
  unsigned int const n_agglomerates = build_agglomerates(agglomerate_dim);

  // Parallel part of the setup.
  std::vector<unsigned int> agglomerate_ids(n_agglomerates);
  std::iota(agglomerate_ids.begin(), agglomerate_ids.end(), 1);
  dealii::WorkStream::run(agglomerate_ids.begin(), agglomerate_ids.end(), *this,
                          &AMGe::local_worker, &AMGe::copy_local_to_global,
                          ScratchData(), CopyData());
}

template <int dim, typename ScalarType>
void AMGe<dim, ScalarType>::local_worker(
    std::vector<unsigned int>::iterator const &agg_id, ScratchData &,
    CopyData &)
{
  dealii::Triangulation<dim> agglomerate_triangulation;
  std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
           typename dealii::DoFHandler<dim>::active_cell_iterator>
      agglomerate_to_global_tria_map;

  std::tie(agglomerate_triangulation, agglomerate_to_global_tria_map) =
      build_agglomerate_triangulation(*agg_id);
}

template <int dim, typename ScalarType>
void AMGe<dim, ScalarType>::copy_local_to_global(CopyData const &)
{
  // do nothing
}

template <int dim, typename ScalarType>
std::vector<dealii::types::global_dof_index>
AMGe<dim, ScalarType>::compute_dof_index_map(
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator> const
        &patch_to_global_map,
    dealii::DoFHandler<dim> const &agglomerate_dof_handler) const
{
  std::vector<dealii::types::global_dof_index> dof_indices(
      agglomerate_dof_handler.n_dofs());

  unsigned int const dofs_per_cell = _dof_handler.get_fe().dofs_per_cell;
  std::vector<dealii::types::global_dof_index> agg_dof_indices(dofs_per_cell);
  std::vector<dealii::types::global_dof_index> global_dof_indices(
      dofs_per_cell);

  for (auto agg_cell : agglomerate_dof_handler.active_cell_iterators())
  {
    agg_cell->get_dof_indices(agg_dof_indices);
    auto global_cell = patch_to_global_map.at(agg_cell);
    global_cell->get_dof_indices(global_dof_indices);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      dof_indices[agg_dof_indices[i]] = global_dof_indices[i];
  }

  return dof_indices;
}

template <int dim, typename ScalarType>
dealii::TrilinosWrappers::SparsityPattern
AMGe<dim, ScalarType>::compute_restriction_sparsity_pattern(
    std::vector<dealii::Vector<double>> const &eigenvectors,
    std::vector<std::vector<dealii::types::global_dof_index>> const
        &dof_indices_maps) const
{
  // Compute the row IndexSet
  int const n_procs = dealii::Utilities::MPI::n_mpi_processes(_comm);
  int const rank = dealii::Utilities::MPI::this_mpi_process(_comm);
  unsigned int const n_local_rows(eigenvectors.size());
  std::vector<unsigned int> n_rows_per_proc(n_procs);
  n_rows_per_proc[rank] = n_local_rows;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &n_rows_per_proc[0], 1,
                MPI_UNSIGNED, _comm);

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
      row_indexset, _dof_handler.locally_owned_dofs(), _comm);

  for (unsigned int i = 0; i < n_local_rows; ++i)
    sp.add_entries(n_rows_before + i, dof_indices_maps[i].begin(),
                   dof_indices_maps[i].end());

  sp.compress();

  return sp;
}
}

#endif
