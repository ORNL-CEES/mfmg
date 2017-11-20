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

#include <mfmg/amge.hpp>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>

namespace mfmg
{
template <int dim, typename VectorType>
AMGe<dim, VectorType>::AMGe(MPI_Comm comm, dealii::DoFHandler<dim> &dof_handler)
    : _comm(comm), _dof_handler(dof_handler)
{
}

template <int dim, typename VectorType>
void AMGe<dim, VectorType>::build_agglomerate(
    std::array<unsigned int, dim> const &agglomerate_dim)
{
  // Faces in deal are orderd as follows: left (x_m) = 0, right (x_p) = 1,
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
      cell->set_user_index(agglomerate);
      auto current_z_cell = cell;
      unsigned int const d_3 = (dim < 3) ? 1 : agglomerate_dim.back();
      for (unsigned int i = 0; i < d_3; ++i)
      {
        auto current_y_cell = current_z_cell;
        for (unsigned int j = 0; j < agglomerate_dim[1]; ++j)
        {
          auto current_cell = current_y_cell;
          for (unsigned int k = 0; k < agglomerate_dim[0]; ++k)
          {
            // TODO For now, we assume that there is no adaptive refine
            current_cell->set_user_index(agglomerate);
            if (current_cell->at_boundary(x_p) == false)
            {
              auto neighbor_cell = current_cell->neighbor(x_p);
              if ((neighbor_cell->is_locally_owned()) &&
                  (neighbor_cell->user_index() == 0))
                current_cell = neighbor_cell;
            }
            else
              break;
          }
          if (current_y_cell->at_boundary(y_p) == false)
          {
            auto neighbor_y_cell = current_y_cell->neighbor(y_p);
            if ((neighbor_y_cell->is_locally_owned()) &&
                (neighbor_y_cell->user_index() == 0))
              current_y_cell = neighbor_y_cell;
          }
          else
            break;
        }
        if ((dim == 3) && (current_z_cell->at_boundary(z_p) == false))
        {
          auto neighbor_z_cell = current_z_cell->neighbor(z_p);
          if ((neighbor_z_cell->is_locally_owned()) &&
              (neighbor_z_cell->user_index() == 0))
            current_z_cell = neighbor_z_cell;
        }
        else
          break;
      }

      ++agglomerate;
    }
  }
}

template <int dim, typename VectorType>
void AMGe<dim, VectorType>::output(std::string const &filename)
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
    std::vector<std::string> full_filenames;
    for (unsigned int i = 0; i < comm_size; ++i)
      full_filenames.push_back(filename + std::to_string(i) + ".vtu");
    std::ofstream master_output(filename + ".pvtu");
    data_out.write_pvtu_record(master_output, full_filenames);
  }
}
}
