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

#define BOOST_TEST_MODULE agglomerate

#include "main.cc"

#include <mfmg/amge.hpp>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/trilinos_vector.h>

#include <array>
#include <boost/mpi.hpp>

BOOST_AUTO_TEST_CASE(simple_agglomerate_2d)
{
  boost::mpi::communicator world;

  dealii::parallel::distributed::Triangulation<2> triangulation(world);
  dealii::FE_Q<2> fe(1);
  dealii::DoFHandler<2> dof_handler(triangulation);

  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(3);
  dof_handler.distribute_dofs(fe);

  mfmg::AMGe<2, dealii::TrilinosWrappers::MPI::Vector> amge(world, dof_handler);

  std::array<unsigned int, 2> agglomerate_dim = {{2, 3}};
  amge.build_agglomerate(agglomerate_dim);

  std::vector<unsigned int> ref_agglomerates;
  std::vector<unsigned int> agglomerates;
  for (auto cell : dof_handler.active_cell_iterators())
    agglomerates.push_back(cell->user_index());
  if (world.size() == 1)
    ref_agglomerates = {{1, 1, 1, 1, 2,  2,  2,  2,  1,  1,  3,  3, 2,
                         2, 4, 4, 5, 5,  5,  5,  6,  6,  6,  6,  5, 5,
                         7, 7, 6, 6, 8,  8,  3,  3,  3,  3,  4,  4, 4,
                         4, 9, 9, 9, 9,  10, 10, 10, 10, 7,  7,  7, 7,
                         8, 8, 8, 8, 11, 11, 11, 11, 12, 12, 12, 12}};
  if (world.size() == 2)
  {
    if (world.rank() == 0)
      ref_agglomerates = {{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2,
                           4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 7, 7, 6, 6, 8, 8,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    else
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 4, 4,
                           5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 7, 7, 6, 6, 8, 8}};
  }
  if (world.size() == 4)
  {
    if (world.rank() == 0)
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                           1, 1, 3, 3, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    if (world.rank() == 1)
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 4,
                           4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    if (world.rank() == 2)
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3,
                           3, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0}};

    if (world.rank() == 3)
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                           1, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 4, 4}};
  }

  BOOST_TEST(agglomerates == ref_agglomerates);
}
