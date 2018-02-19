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

#ifndef AMGE_HPP
#define AMGE_HPP

#include <deal.II/base/mpi.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <array>
#include <map>
#include <string>

namespace mfmg
{
template <int dim, typename VectorType>
class AMGe
{
public:
  AMGe(MPI_Comm comm, dealii::DoFHandler<dim> const &dof_handler);

  /**
   * Flag cells to create agglomerates. The desired size of the agglomerates is
   * given by \p agglomerate_dim. This functions returns the number of
   * agglomerates that have been created.
   */
  unsigned int build_agglomerates(
      std::array<unsigned int, dim> const &agglomerate_dim) const;

  /**
   * Create a Triangulation \p agglomerate_triangulation associated with an
   * agglomerate of a given \p agglomerate_id and a map that matches cells in
   * the local triangulation with cells in the global triangulation.
   */
  void build_agglomerate_triangulation(
      unsigned int agglomerate_id,
      dealii::Triangulation<dim> &agglomerate_triangulation,
      std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
               typename dealii::DoFHandler<dim>::active_cell_iterator>
          &agglomerate_to_global_tria_map) const;

  /**
   * Compute the map between the dof indices of the local DoFHandler and the
   * dof indices of the global DoFHandler.
   */
  std::vector<dealii::types::global_dof_index> compute_dof_index_map(
      std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
               typename dealii::DoFHandler<dim>::active_cell_iterator> const
          &patch_to_global_map,
      dealii::DoFHandler<dim> const &agglomerate_dof_handler) const;

  /**
   * Output the mesh and the agglomerate ids.
   */
  void output(std::string const &filename) const;

protected:
  MPI_Comm _comm;
  dealii::DoFHandler<dim> const &_dof_handler;
};
}

#endif
