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

#include <deal.II/base/mpi.h>
#include <deal.II/dofs/dof_handler.h>

#include <array>
#include <map>
#include <string>

namespace mfmg
{
template <int dim, typename VectorType>
class AMGe
{
public:
  AMGe(MPI_Comm comm, dealii::DoFHandler<dim> &dof_handler);

  unsigned int
  build_agglomerates(std::array<unsigned int, dim> const &agglomerate_dim);

  void build_agglomerate_triangulation(
      unsigned int agglomerate_id,
      dealii::Triangulation<dim> &agglomerate_triangulation,
      std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
               typename dealii::DoFHandler<dim>::active_cell_iterator>
          &agglomerate_to_global_tria_map);

  void output(std::string const &filename);

  void setup(std::array<unsigned int, dim> const &agglomerate_dim);

private:
  struct ScratchData
  {
    // nothing
  };

  struct CopyData
  {
    // nothing
  };

  void local_worker(std::vector<unsigned int>::iterator const &agg_id,
                    ScratchData &scratch_data, CopyData &copy_data);

  void copy_local_to_global(CopyData const &copy_data);

  MPI_Comm _comm;
  dealii::DoFHandler<dim> &_dof_handler;
};
}
