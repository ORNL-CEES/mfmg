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
#include <deal.II/numerics/data_out.h>

#include <fstream>

namespace mfmg
{
template <int dim, typename NumberType>
AMGe<dim, NumberType>::AMGe(MPI_Comm comm,
                            dealii::DoFHandler<dim> const &dof_handler)
    : _comm(comm), _dof_handler(dof_handler)
{
}

template <int dim, typename NumberType>
unsigned int AMGe<dim, NumberType>::build_agglomerates(
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

template <int dim, typename NumberType>
std::tuple<dealii::Triangulation<dim>,
           std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
                    typename dealii::DoFHandler<dim>::active_cell_iterator>>
AMGe<dim, NumberType>::build_agglomerate_triangulation(
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

template <int dim, typename NumberType>
void AMGe<dim, NumberType>::output(std::string const &filename) const
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

template <int dim, typename NumberType>
void AMGe<dim, NumberType>::setup(
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

template <int dim, typename NumberType>
void AMGe<dim, NumberType>::local_worker(
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

template <int dim, typename NumberType>
void AMGe<dim, NumberType>::copy_local_to_global(CopyData const &)
{
  // do nothing
}
}

#endif
