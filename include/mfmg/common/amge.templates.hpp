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

#ifndef AMG_TEMPLATES_HPP
#define AMG_TEMPLATES_HPP

#include <mfmg/common/amge.hpp>
#include <mfmg/common/exceptions.hpp>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <unordered_map>

#ifdef DEAL_II_TRILINOS_WITH_ZOLTAN
// Zoltan random seed control is in an internal zz_rand.h file which is not
// installed with Trilinos. Thus, we duplicate the signature and the
// initialization value here.
#define ZOLTAN_RAND_INIT 123456789U
extern "C"
{
  extern void Zoltan_Srand(unsigned int, unsigned int *);
}
#endif

namespace mfmg
{
template <int dim, typename VectorType>
AMGe<dim, VectorType>::AMGe(MPI_Comm comm,
                            dealii::DoFHandler<dim> const &dof_handler)
    : _comm(comm), _dof_handler(dof_handler)
{
}

template <int dim, typename VectorType>
unsigned int AMGe<dim, VectorType>::build_agglomerates(
    boost::property_tree::ptree const &ptree) const
{
  std::string partitioner_type = ptree.get<std::string>("partitioner");
  std::transform(partitioner_type.begin(), partitioner_type.end(),
                 partitioner_type.begin(), ::tolower);
  if ((partitioner_type == "zoltan") || (partitioner_type == "metis"))
  {
#ifdef DEAL_II_TRILINOS_WITH_ZOLTAN
    // Always use the same seed for Zoltan
    Zoltan_Srand(ZOLTAN_RAND_INIT, NULL);
#endif
    unsigned int const n_desired_agglomerates =
        ptree.get<unsigned int>("n_agglomerates");

    return build_agglomerates_partitioner(partitioner_type,
                                          n_desired_agglomerates);
  }
  else if (partitioner_type == "block")
  {
    std::array<unsigned int, dim> agglomerate_dim;
    agglomerate_dim[0] = ptree.get<unsigned int>("nx");
    agglomerate_dim[1] = ptree.get<unsigned int>("ny");
    if (dim == 3)
      agglomerate_dim[2] = ptree.get<unsigned int>("nz");

    return build_agglomerates_block(agglomerate_dim);
  }
  else
    ASSERT_THROW(false, partitioner_type +
                            " is not a valid choice for the partitioner. The "
                            "acceptable values are zoltan, metis, and block.");
  return 0;
}

template <int dim, typename VectorType>
std::pair<std::vector<std::vector<unsigned int>>,
          std::vector<std::vector<unsigned int>>>
AMGe<dim, VectorType>::build_boundary_agglomerates() const
{
  auto filtered_iterators_range =
      filter_iterators(_dof_handler.active_cell_iterators(),
                       dealii::IteratorFilters::LocallyOwnedCell());
  std::vector<std::vector<unsigned int>> agg_cell_id(_n_agglomerates);
  std::vector<std::set<unsigned int>> agg_cell_set(_n_agglomerates);
  for (auto cell : filtered_iterators_range)
  {
    agg_cell_id[cell->user_index() - 1].push_back(cell->active_cell_index());
    agg_cell_set[cell->user_index() - 1].insert(cell->active_cell_index());
  }

  dealii::DynamicSparsityPattern connectivity;
  dealii::GridTools::get_vertex_connectivity_of_cells(
      _dof_handler.get_triangulation(), connectivity);

  // TODO do this using multithreading. Maybe it's better to do everything in
  // the following for loop
  // Each agglomerate will create two new agglomerates: one composed of the
  // cells of the agglomerate which are on the boundary with another agglomerate
  // and another one composed of cells on other agglomerates that share a
  // boundary with the current agglomerate
  std::vector<std::vector<unsigned int>> interior_agglomerates(_n_agglomerates);
  std::vector<std::vector<unsigned int>> halo_agglomerates(_n_agglomerates);
  for (unsigned int i = 0; i < _n_agglomerates; ++i)
  {
    unsigned int const n_cells_agg = agg_cell_id[i].size();
    std::vector<unsigned int> interior_boundary_cells;
    std::vector<unsigned int> halo_cells;
    std::set<unsigned int> halo_cells_in_agg;
    for (unsigned int j = 0; j < n_cells_agg; ++j)
    {
      bool cell_in_agg = false;
      // Get the connectivity for the current cell
      auto connectivity_begin = connectivity.begin(agg_cell_id[i][j]);
      auto connectivity_end = connectivity.end(agg_cell_id[i][j]);
      for (auto connectivity_it = connectivity_begin;
           connectivity_it != connectivity_end; ++connectivity_it)
      {
        // Cells that are on the boundary of agglomerates and have in their
        // connectivity cells that are not part of the agglomerates
        if (agg_cell_set[i].count(connectivity_it->column()) == 0)
        {
          if (cell_in_agg == false)
          {
            interior_boundary_cells.push_back(agg_cell_id[i][j]);
            cell_in_agg = true;
          }
          if (halo_cells_in_agg.count(connectivity_it->column()) == 0)
          {
            halo_cells.push_back(connectivity_it->column());
            halo_cells_in_agg.insert(connectivity_it->column());
          }
        }
      }
    }

    interior_agglomerates[i] = interior_boundary_cells;
    halo_agglomerates[i] = halo_cells;
  }

  return {interior_agglomerates, halo_agglomerates};
}

template <int dim, typename VectorType>
void AMGe<dim, VectorType>::build_agglomerate_triangulation(
    unsigned int agglomerate_id,
    dealii::Triangulation<dim> &agglomerate_triangulation,
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator>
        &agglomerate_to_global_tria_map) const
{
  std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>
      agglomerate;
  for (auto cell : _dof_handler.active_cell_iterators())
    if (cell->user_index() == agglomerate_id)
      agglomerate.push_back(cell);

  build_agglomerate_triangulation(agglomerate, agglomerate_triangulation,
                                  agglomerate_to_global_tria_map);
}

template <int dim, typename VectorType>
void AMGe<dim, VectorType>::build_agglomerate_triangulation(
    std::vector<unsigned int> const &cell_index,
    dealii::Triangulation<dim> &agglomerate_triangulation,
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator>
        &agglomerate_to_global_tria_map) const
{
  std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>
      agglomerate;
  for (auto cell_id : cell_index)
  {
    auto cell = _dof_handler.begin_active();
    std::advance(cell, cell_id);
    agglomerate.push_back(cell);
  }

  build_agglomerate_triangulation(agglomerate, agglomerate_triangulation,
                                  agglomerate_to_global_tria_map);
}

template <int dim, typename VectorType>
std::vector<dealii::types::global_dof_index>
AMGe<dim, VectorType>::compute_dof_index_map(
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

template <int dim, typename VectorType>
void AMGe<dim, VectorType>::output(std::string const &filename) const
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

template <int dim, typename VectorType>
void AMGe<dim, VectorType>::compute_restriction_sparse_matrix(
    std::vector<dealii::Vector<typename VectorType::value_type>> const
        &eigenvectors,
    std::vector<std::vector<typename VectorType::value_type>> const
        &diag_elements,
    std::vector<std::vector<dealii::types::global_dof_index>> const
        &dof_indices_maps,
    std::vector<unsigned int> const &n_local_eigenvectors,
    dealii::LinearAlgebra::distributed::Vector<
        typename VectorType::value_type> const &locally_relevant_global_diag,
    dealii::TrilinosWrappers::SparseMatrix &restriction_sparse_matrix) const
{
  // Compute the sparsity pattern (Epetra_FECrsGraph)
  dealii::TrilinosWrappers::SparsityPattern restriction_sp =
      compute_restriction_sparsity_pattern(eigenvectors, dof_indices_maps,
                                           n_local_eigenvectors);

  // Build the restriction sparse matrix
  restriction_sparse_matrix.reinit(restriction_sp);
  std::pair<dealii::types::global_dof_index,
            dealii::types::global_dof_index> const local_range =
      restriction_sp.local_range();
  unsigned int const n_agglomerates = n_local_eigenvectors.size();
  unsigned int pos = 0;
  ASSERT(n_agglomerates == dof_indices_maps.size(),
         "dof_indices_maps has the wrong size: " +
             std::to_string(dof_indices_maps.size()) + " instead of " +
             std::to_string(n_agglomerates));
  for (unsigned int i = 0; i < n_agglomerates; ++i)
  {
    unsigned int const n_local_eig = n_local_eigenvectors[i];
    for (unsigned int k = 0; k < n_local_eig; ++k)
    {
      unsigned int const n_elem = eigenvectors[pos].size();
      ASSERT(n_elem == dof_indices_maps[i].size(),
             "dof_indices_maps[i] has the wrong size: " +
                 std::to_string(dof_indices_maps[i].size()) + " instead of " +
                 std::to_string(n_elem));
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
  // Check that the locally_relevant_global_diag is the sum of the agglomerates
  // diagonal
  // TODO do not ask user for the locally_relevant_global_diag
  dealii::LinearAlgebra::distributed::Vector<typename VectorType::value_type>
      new_global_diag(locally_relevant_global_diag.get_partitioner());
  for (unsigned int i = 0; i < n_agglomerates; ++i)
  {
    // Get the size of the eigenvectors in agglomerate i
    unsigned int offset = std::accumulate(n_local_eigenvectors.begin(),
                                          n_local_eigenvectors.begin() + i, 0);
    unsigned int const n_elem = eigenvectors[offset].size();

    for (unsigned int j = 0; j < n_elem; ++j)
    {
      dealii::types::global_dof_index const global_pos = dof_indices_maps[i][j];
      new_global_diag[global_pos] += diag_elements[i][j];
    }
  }
  new_global_diag.compress(dealii::VectorOperation::add);
  new_global_diag -= locally_relevant_global_diag;
  ASSERT((new_global_diag.linfty_norm() /
          locally_relevant_global_diag.linfty_norm()) < 1e-14,
         "Sum of agglomerate diagonals is not equal to the global diagonal");

  // Check that the sum of the weight matrices is the identity
  auto locally_owned_dofs =
      locally_relevant_global_diag.locally_owned_elements();
  dealii::TrilinosWrappers::SparsityPattern sp(locally_owned_dofs,
                                               locally_owned_dofs, this->_comm);
  for (auto local_index : locally_owned_dofs)
    sp.add(local_index, local_index);
  sp.compress();

  dealii::TrilinosWrappers::SparseMatrix weight_matrix(sp);
  pos = 0;
  for (unsigned int i = 0; i < n_agglomerates; ++i)
  {
    unsigned int const n_elem = eigenvectors[pos].size();
    for (unsigned int j = 0; j < n_elem; ++j)
    {
      dealii::types::global_dof_index const global_pos = dof_indices_maps[i][j];
      double const value =
          diag_elements[i][j] / locally_relevant_global_diag[global_pos];
      weight_matrix.add(global_pos, global_pos, value);
    }
    pos += n_local_eigenvectors[i];
  }

  // Compress the matrix
  weight_matrix.compress(dealii::VectorOperation::add);

  for (auto index : locally_owned_dofs)
    ASSERT(std::abs(weight_matrix.diag_element(index) - 1.0) < 1e-14,
           "Sum of local weight matrices is not the identity");
#endif
}

template <int dim, typename VectorType>
void AMGe<dim, VectorType>::compute_restriction_sparse_matrix(
    std::vector<typename VectorType::value_type> const &eigenvalues,
    std::vector<dealii::Vector<typename VectorType::value_type>> const
        &eigenvectors,
    std::vector<std::vector<typename VectorType::value_type>> const
        &diag_elements,
    std::vector<std::vector<dealii::types::global_dof_index>> const
        &dof_indices_maps,
    std::vector<unsigned int> const &n_local_eigenvectors,
    dealii::LinearAlgebra::distributed::Vector<
        typename VectorType::value_type> const &locally_relevant_global_diag,
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix>
        restriction_sparse_matrix,
    std::unique_ptr<dealii::TrilinosWrappers::SparseMatrix>
        &eigenvector_sparse_matrix,
    std::unique_ptr<dealii::TrilinosWrappers::SparseMatrix>
        &delta_eigenvector_matrix) const
{
  // Compute the sparsity pattern (Epetra_FECrsGraph)
  dealii::TrilinosWrappers::SparsityPattern restriction_sp =
      compute_restriction_sparsity_pattern(eigenvectors, dof_indices_maps,
                                           n_local_eigenvectors);

  // Build the sparse matrices
  restriction_sparse_matrix->reinit(restriction_sp);
  eigenvector_sparse_matrix->reinit(restriction_sp);
  // The sparsity pattern is different than for the other sparse matrices
  // because some of the entries that do not correspond to agglomerate boundary
  // are zeros. Because reinit requires the SparsityPattern which is harder to
  // compute, we instead use the constructor that computes the SparsityPattern
  // when compress() is called).
  delta_eigenvector_matrix.reset(new dealii::TrilinosWrappers::SparseMatrix(
      eigenvector_sparse_matrix->locally_owned_range_indices(),
      eigenvector_sparse_matrix->locally_owned_domain_indices(),
      eigenvector_sparse_matrix->get_mpi_communicator()));

  std::pair<dealii::types::global_dof_index,
            dealii::types::global_dof_index> const local_range =
      restriction_sp.local_range();
  unsigned int const n_agglomerates = n_local_eigenvectors.size();
  unsigned int pos = 0;
  ASSERT(n_agglomerates == dof_indices_maps.size(),
         "dof_indices_maps has the wrong size: " +
             std::to_string(dof_indices_maps.size()) + " instead of " +
             std::to_string(n_agglomerates));
  for (unsigned int i = 0; i < n_agglomerates; ++i)
  {
    unsigned int const n_local_eig = n_local_eigenvectors[i];
    for (unsigned int k = 0; k < n_local_eig; ++k)
    {
      unsigned int const n_elem = eigenvectors[pos].size();
      ASSERT(n_elem == dof_indices_maps[i].size(),
             "dof_indices_maps[i] has the wrong size: " +
                 std::to_string(dof_indices_maps[i].size()) + " instead of " +
                 std::to_string(n_elem));
      for (unsigned int j = 0; j < n_elem; ++j)
      {
        dealii::types::global_dof_index const global_pos =
            dof_indices_maps[i][j];
        // Fill restriction sparse matrix
        restriction_sparse_matrix->add(
            local_range.first + pos, global_pos,
            diag_elements[i][j] / locally_relevant_global_diag[global_pos] *
                eigenvectors[pos][j]);
        // Fill eigenvector sparse matrix
        eigenvector_sparse_matrix->add(local_range.first + pos, global_pos,
                                       eigenvalues[pos] * eigenvectors[pos][j]);
        // Fill delta eigenvector sparse matrix
        delta_eigenvector_matrix->set(
            local_range.first + pos, global_pos,
            (diag_elements[i][j] / locally_relevant_global_diag[global_pos] -
             1.) *
                eigenvectors[pos][j]);
      }
      ++pos;
    }
  }

  // Compress the matrices
  restriction_sparse_matrix->compress(dealii::VectorOperation::add);
  eigenvector_sparse_matrix->compress(dealii::VectorOperation::add);
  delta_eigenvector_matrix->compress(dealii::VectorOperation::insert);

#if MFMG_DEBUG
  // Check that the sum of the weight matrices is the identity
  auto locally_owned_dofs =
      locally_relevant_global_diag.locally_owned_elements();
  dealii::TrilinosWrappers::SparsityPattern sp(locally_owned_dofs,
                                               locally_owned_dofs, this->_comm);
  for (auto local_index : locally_owned_dofs)
    sp.add(local_index, local_index);
  sp.compress();

  dealii::TrilinosWrappers::SparseMatrix weight_matrix(sp);
  pos = 0;
  for (unsigned int i = 0; i < n_agglomerates; ++i)
  {
    unsigned int const n_elem = eigenvectors[pos].size();
    for (unsigned int j = 0; j < n_elem; ++j)
    {
      dealii::types::global_dof_index const global_pos = dof_indices_maps[i][j];
      double const value =
          diag_elements[i][j] / locally_relevant_global_diag[global_pos];
      weight_matrix.add(global_pos, global_pos, value);
    }
    pos += n_local_eigenvectors[i];
  }

  // Compress the matrix
  weight_matrix.compress(dealii::VectorOperation::add);

  for (auto index : locally_owned_dofs)
    ASSERT(std::abs(weight_matrix.diag_element(index) - 1.0) < 1e-14,
           "Sum of local weight matrices is not the identity");
#endif
}

template <int dim, typename VectorType>
unsigned int AMGe<dim, VectorType>::build_agglomerates_block(
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

  _n_agglomerates = agglomerate - 1;

  return _n_agglomerates;
}

template <int dim, typename VectorType>
unsigned int AMGe<dim, VectorType>::build_agglomerates_partitioner(
    std::string const &partitioner_type, unsigned int n_agglomerates) const
{
  // We cannot use deal.II wrappers to create the agglomerates because
  //   1) the wrappers only works on serial Triangulation
  //   2) they override the subdomain_id which is already used by p4est
  // Instead, we create the connectivity graph ourselves and then, we do the
  // partitioning.

  unsigned int const n_local_cells =
      _dof_handler.get_triangulation().n_active_cells();

  // Create the DynamicSparsityPattern
  dealii::DynamicSparsityPattern connectivity(n_local_cells);

  // Associate a local index to each cell
  unsigned int local_index = 0;
  std::map<std::pair<unsigned int, unsigned int>, unsigned int> index_map;
  for (auto cell : _dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      index_map[std::make_pair(cell->level(), cell->index())] = local_index;
      ++local_index;
    }
  }

  // Fill the dynamic connectivity sparsity pattern
  for (auto cell : _dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      unsigned int const index =
          index_map.at(std::make_pair(cell->level(), cell->index()));
      connectivity.add(index, index);
      for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell;
           ++f)
      {
        if ((cell->at_boundary(f) == false) &&
            (cell->neighbor(f)->is_locally_owned() == true) &&
            (cell->neighbor(f)->has_children() == false))
        {
          unsigned int const neighbor_index = index_map.at(std::make_pair(
              cell->neighbor(f)->level(), cell->neighbor(f)->index()));
          connectivity.add(index, neighbor_index);
          connectivity.add(neighbor_index, index);
        }
      }
    }
  }

  dealii::SparsityPattern cell_connectivity;
  cell_connectivity.copy_from(connectivity);

  // Partition the connection graph
  dealii::SparsityTools::Partitioner partitioner;
  if (partitioner_type == "metis")
    partitioner = dealii::SparsityTools::Partitioner::metis;
  else
    partitioner = dealii::SparsityTools::Partitioner::zoltan;
  std::vector<unsigned int> partition_indices(n_local_cells);
  dealii::SparsityTools::partition(cell_connectivity, n_agglomerates,
                                   partition_indices, partitioner);

  // Assign the agglomerate ID to all the locally owned cells. Zoltan does not
  // guarantee that the agglomerate IDs will consecutive so we need to
  // renumber them. The lowest agglomerate ID is one because zero is reserved
  // for ghost and artificial cells.
  unsigned int n_zoltan_agglomerates = 0;
  std::unordered_map<unsigned int, unsigned int> agglomerate_renumbering;
  for (auto cell : _dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      unsigned int const index =
          index_map.at(std::make_pair(cell->level(), cell->index()));
      auto agg_id = agglomerate_renumbering.find(partition_indices[index]);
      if (agg_id == agglomerate_renumbering.end())
      {
        ++n_zoltan_agglomerates;
        agglomerate_renumbering[partition_indices[index]] =
            n_zoltan_agglomerates;
        cell->set_user_index(n_zoltan_agglomerates);
      }
      else
        cell->set_user_index(agg_id->second);
    }
  }

  _n_agglomerates = n_zoltan_agglomerates;

  return _n_agglomerates;
}

template <int dim, typename VectorType>
dealii::TrilinosWrappers::SparsityPattern
AMGe<dim, VectorType>::compute_restriction_sparsity_pattern(
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

template <int dim, typename VectorType>
void AMGe<dim, VectorType>::build_agglomerate_triangulation(
    std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator> const
        &agglomerate,
    dealii::Triangulation<dim> &agglomerate_triangulation,
    std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
             typename dealii::DoFHandler<dim>::active_cell_iterator>
        &agglomerate_to_global_tria_map) const
{
  // Map between the cells on the boundary and the faces on the boundary and the
  // associated boundary id.
  std::map<typename dealii::DoFHandler<dim>::active_cell_iterator,
           std::vector<std::pair<unsigned int, unsigned int>>>
      boundary_ids;
  for (auto cell : agglomerate)
    if (cell->at_boundary())
    {
      for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell;
           ++f)
      {
        if (cell->face(f)->at_boundary())
        {
          boundary_ids[cell].push_back(
              std::make_pair(f, cell->face(f)->boundary_id()));
        }
      }
    }

  // If the agglomerate has hanging nodes, the patch is bigger than
  // what we may expect because we cannot create a coarse triangulation with
  // hanging nodes. Thus, we need to use FE_Nothing to get ride of unwanted
  // cells.
  dealii::GridTools::build_triangulation_from_patch<dealii::DoFHandler<dim>>(
      agglomerate, agglomerate_triangulation, agglomerate_to_global_tria_map);

  // Copy the boundary IDs to the agglomerate triangulation
  for (auto const &boundary : boundary_ids)
  {
    auto const boundary_cell = boundary.first;
    for (auto &agglomerate_cell : agglomerate_to_global_tria_map)
    {
      if (agglomerate_cell.second == boundary_cell)
      {
        for (auto &boundary_face : boundary.second)
          agglomerate_cell.first->face(boundary_face.first)
              ->set_boundary_id(boundary_face.second);

        break;
      }
    }
  }
}
} // namespace mfmg

#endif
