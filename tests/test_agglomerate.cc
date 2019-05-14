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

#define BOOST_TEST_MODULE agglomerate

#include <mfmg/dealii/amge_host.hpp>
#include <mfmg/dealii/dealii_mesh_evaluator.hpp>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/data_out.h>

#include <array>

#include "main.cc"

template <int dim>
std::pair<std::vector<unsigned int>,
          std::pair<std::vector<std::vector<unsigned int>>,
                    std::vector<std::vector<unsigned int>>>>
test(MPI_Comm const &world, bool const &boundary = false)
{
  dealii::parallel::distributed::Triangulation<dim> triangulation(world);
  dealii::FE_Q<dim> fe(1);
  dealii::DoFHandler<dim> dof_handler(triangulation);

  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(3);
  dof_handler.distribute_dofs(fe);

  using Vector = dealii::LinearAlgebra::distributed::Vector<double>;
  using DummyMeshEvaluator = mfmg::DealIIMeshEvaluator<dim>;

  mfmg::AMGe_host<dim, DummyMeshEvaluator, Vector> amge(world, dof_handler);

  boost::property_tree::ptree partitioner_params;
  partitioner_params.put("partitioner", "block");
  partitioner_params.put("nx", 2);
  partitioner_params.put("ny", 3);
  partitioner_params.put("nz", 4);

  amge.build_agglomerates(partitioner_params);

  std::vector<unsigned int> agglomerates;
  agglomerates.reserve(dof_handler.get_triangulation().n_active_cells());
  for (auto cell : dof_handler.active_cell_iterators())
    agglomerates.push_back(cell->user_index());

  std::pair<std::vector<std::vector<unsigned int>>,
            std::vector<std::vector<unsigned int>>>
      boundary_agglomerates;
  if (boundary)
    boundary_agglomerates = amge.build_boundary_agglomerates();

  return std::make_pair(agglomerates, boundary_agglomerates);
}

BOOST_AUTO_TEST_CASE(simple_agglomerate_2d)
{
  std::vector<unsigned int> agglomerates;
  std::tie(agglomerates, std::ignore) = test<2>(MPI_COMM_WORLD);

  unsigned int world_size =
      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  unsigned int world_rank =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  std::vector<unsigned int> ref_agglomerates;
  if (world_size == 1)
    ref_agglomerates = {{1, 1, 1, 1, 2,  2,  2,  2,  1,  1,  3,  3, 2,
                         2, 4, 4, 5, 5,  5,  5,  6,  6,  6,  6,  5, 5,
                         7, 7, 6, 6, 8,  8,  3,  3,  3,  3,  4,  4, 4,
                         4, 9, 9, 9, 9,  10, 10, 10, 10, 7,  7,  7, 7,
                         8, 8, 8, 8, 11, 11, 11, 11, 12, 12, 12, 12}};
  else if (world_size == 2)
  {
    if (world_rank == 0)
      ref_agglomerates = {{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2,
                           4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 7, 7, 6, 6, 8, 8,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    else
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 4, 4,
                           5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 7, 7, 6, 6, 8, 8}};
  }
  else if (world_size == 4)
  {
    if (world_rank == 0)
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                           1, 1, 3, 3, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    if (world_rank == 1)
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 4,
                           4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    if (world_rank == 2)
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3,
                           3, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0}};

    if (world_rank == 3)
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                           1, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 4, 4}};
  }

  BOOST_TEST(agglomerates == ref_agglomerates);
}

BOOST_AUTO_TEST_CASE(simple_agglomerate_3d)
{
  unsigned int world_size =
      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  unsigned int world_rank =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  std::vector<unsigned int> agglomerates;
  std::tie(agglomerates, std::ignore) = test<3>(MPI_COMM_WORLD);
  std::vector<unsigned int> ref_agglomerates;

  if (world_size == 1)
  {
    ref_agglomerates = {
        {1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  1,  1,
         3,  3,  1,  1,  3,  3,  2,  2,  4,  4,  2,  2,  4,  4,  1,  1,  1,  1,
         1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  1,  1,  3,  3,  1,  1,
         3,  3,  2,  2,  4,  4,  2,  2,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,
         6,  6,  6,  6,  6,  6,  6,  6,  5,  5,  7,  7,  5,  5,  7,  7,  6,  6,
         8,  8,  6,  6,  8,  8,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,
         6,  6,  6,  6,  5,  5,  7,  7,  5,  5,  7,  7,  6,  6,  8,  8,  6,  6,
         8,  8,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,
         9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 3,  3,
         3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  9,  9,  9,  9,
         9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 7,  7,  7,  7,  7,  7,
         7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  11, 11, 11, 11, 11, 11, 11, 11,
         12, 12, 12, 12, 12, 12, 12, 12, 7,  7,  7,  7,  7,  7,  7,  7,  8,  8,
         8,  8,  8,  8,  8,  8,  11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12,
         12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14,
         14, 14, 13, 13, 15, 15, 13, 13, 15, 15, 14, 14, 16, 16, 14, 14, 16, 16,
         13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13,
         15, 15, 13, 13, 15, 15, 14, 14, 16, 16, 14, 14, 16, 16, 17, 17, 17, 17,
         17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 17, 17, 19, 19, 17, 17,
         19, 19, 18, 18, 20, 20, 18, 18, 20, 20, 17, 17, 17, 17, 17, 17, 17, 17,
         18, 18, 18, 18, 18, 18, 18, 18, 17, 17, 19, 19, 17, 17, 19, 19, 18, 18,
         20, 20, 18, 18, 20, 20, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16,
         16, 16, 16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
         22, 22, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16,
         21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 19, 19,
         19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 23, 23, 23, 23,
         23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 19, 19, 19, 19, 19, 19,
         19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 23, 23, 23, 23, 23, 23, 23, 23,
         24, 24, 24, 24, 24, 24, 24, 24}};
  }
  else if (world_size == 2)
  {
    if (world_rank == 0)
      ref_agglomerates = {
          {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
           1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  1,  1,
           3,  3,  1,  1,  3,  3,  2,  2,  4,  4,  2,  2,  4,  4,  1,  1,  1,
           1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  1,  1,  3,  3,
           1,  1,  3,  3,  2,  2,  4,  4,  2,  2,  4,  4,  5,  5,  5,  5,  5,
           5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  5,  5,  7,  7,  5,  5,
           7,  7,  6,  6,  8,  8,  6,  6,  8,  8,  5,  5,  5,  5,  5,  5,  5,
           5,  6,  6,  6,  6,  6,  6,  6,  6,  5,  5,  7,  7,  5,  5,  7,  7,
           6,  6,  8,  8,  6,  6,  8,  8,  3,  3,  3,  3,  3,  3,  3,  3,  4,
           4,  4,  4,  4,  4,  4,  4,  9,  9,  9,  9,  9,  9,  9,  9,  10, 10,
           10, 10, 10, 10, 10, 10, 3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,
           4,  4,  4,  4,  4,  9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10,
           10, 10, 10, 10, 7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,
           8,  8,  8,  11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12,
           12, 12, 7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,
           8,  11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0}};
    else
      ref_agglomerates = {
          {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,
           2,  2,  2,  2,  2,  2,  2,  1,  1,  3,  3,  1,  1,  3,  3,  2,  2,
           4,  4,  2,  2,  4,  4,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,
           2,  2,  2,  2,  2,  1,  1,  3,  3,  1,  1,  3,  3,  2,  2,  4,  4,
           2,  2,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,
           6,  6,  6,  5,  5,  7,  7,  5,  5,  7,  7,  6,  6,  8,  8,  6,  6,
           8,  8,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,
           6,  5,  5,  7,  7,  5,  5,  7,  7,  6,  6,  8,  8,  6,  6,  8,  8,
           3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  9,
           9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 3,  3,
           3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  9,  9,  9,
           9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 7,  7,  7,  7,
           7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  11, 11, 11, 11, 11,
           11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 7,  7,  7,  7,  7,  7,
           7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  11, 11, 11, 11, 11, 11, 11,
           11, 12, 12, 12, 12, 12, 12, 12, 12}};
  }
  else if (world_size == 4)
  {
    if (world_rank == 0)
      ref_agglomerates = {
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,
           3, 3, 1, 1, 3, 3, 2, 2, 4, 4, 2, 2, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2,
           2, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 1, 1, 3, 3, 2, 2, 4, 4, 2, 2, 4, 4,
           5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 7, 7, 5, 5, 7,
           7, 6, 6, 8, 8, 6, 6, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
           6, 6, 5, 5, 7, 7, 5, 5, 7, 7, 6, 6, 8, 8, 6, 6, 8, 8, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    if (world_rank == 1)
      ref_agglomerates = {
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 1, 1, 3,
           3, 2, 2, 4, 4, 2, 2, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
           2, 2, 1, 1, 3, 3, 1, 1, 3, 3, 2, 2, 4, 4, 2, 2, 4, 4, 5, 5, 5, 5, 5,
           5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 7, 7, 5, 5, 7, 7, 6, 6, 8, 8,
           6, 6, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 7,
           7, 5, 5, 7, 7, 6, 6, 8, 8, 6, 6, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    if (world_rank == 2)
      ref_agglomerates = {
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
           2, 2, 1, 1, 3, 3, 1, 1, 3, 3, 2, 2, 4, 4, 2, 2, 4, 4, 1, 1, 1, 1, 1,
           1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 1, 1, 3, 3, 2, 2, 4, 4,
           2, 2, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 7,
           7, 5, 5, 7, 7, 6, 6, 8, 8, 6, 6, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6,
           6, 6, 6, 6, 6, 6, 5, 5, 7, 7, 5, 5, 7, 7, 6, 6, 8, 8, 6, 6, 8, 8, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    if (world_rank == 3)
      ref_agglomerates = {
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 3,
           3, 1, 1, 3, 3, 2, 2, 4, 4, 2, 2, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
           2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 1, 1, 3, 3, 2, 2, 4, 4, 2, 2, 4, 4, 5,
           5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 7, 7, 5, 5, 7, 7,
           6, 6, 8, 8, 6, 6, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6,
           6, 5, 5, 7, 7, 5, 5, 7, 7, 6, 6, 8, 8, 6, 6, 8, 8}};
  }

  BOOST_TEST(agglomerates == ref_agglomerates);
}

BOOST_AUTO_TEST_CASE(zoltan_agglomerate_2d)
{
  int constexpr dim = 2;
  unsigned int world_size =
      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  unsigned int world_rank =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  dealii::parallel::distributed::Triangulation<dim> triangulation(
      MPI_COMM_WORLD);
  dealii::FE_Q<dim> fe(1);
  dealii::DoFHandler<dim> dof_handler(triangulation);

  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(3);
  dof_handler.distribute_dofs(fe);

  using Vector = dealii::LinearAlgebra::distributed::Vector<double>;
  using DummyMeshEvaluator = mfmg::DealIIMeshEvaluator<dim>;

  mfmg::AMGe_host<dim, DummyMeshEvaluator, Vector> amge(MPI_COMM_WORLD,
                                                        dof_handler);

  boost::property_tree::ptree partitioner_params;
  partitioner_params.put("partitioner", "zoltan");
  partitioner_params.put("n_agglomerates", 3);
  amge.build_agglomerates(partitioner_params);

  std::vector<unsigned int> agglomerates;
  agglomerates.reserve(dof_handler.get_triangulation().n_active_cells());
  for (auto cell : dof_handler.active_cell_iterators())
    agglomerates.push_back(cell->user_index());

  std::vector<unsigned int> ref_agglomerates;
  if (world_size == 1)
  {
    ref_agglomerates = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                        1, 1, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                        2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
  }
  else if (world_size == 2)
  {
    if (world_rank == 0)
      ref_agglomerates = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2,
                          1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    else
      ref_agglomerates = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  }
  else if (world_size == 4)
  {
    if (world_rank == 0)
      ref_agglomerates = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (world_rank == 1)
      ref_agglomerates = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (world_rank == 2)
      ref_agglomerates = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0};
    if (world_rank == 3)
      ref_agglomerates = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2};
  }

  BOOST_TEST(agglomerates == ref_agglomerates);
}

BOOST_AUTO_TEST_CASE(boundary_agglomerate_2d)
{
  bool const boundary = true;
  std::vector<unsigned int> agglomerates;
  std::pair<std::vector<std::vector<unsigned int>>,
            std::vector<std::vector<unsigned int>>>
      boundary_agglomerates;
  std::tie(agglomerates, boundary_agglomerates) =
      test<2>(MPI_COMM_WORLD, boundary);

  unsigned int world_size =
      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  unsigned int world_rank =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  std::vector<unsigned int> ref_agglomerates;
  std::vector<std::vector<unsigned int>> ref_interior_agglomerates;
  std::vector<std::vector<unsigned int>> ref_halo_agglomerates;
  if (world_size == 1)
  {
    ref_agglomerates = {{1, 1, 1, 1, 2,  2,  2,  2,  1,  1,  3,  3, 2,
                         2, 4, 4, 5, 5,  5,  5,  6,  6,  6,  6,  5, 5,
                         7, 7, 6, 6, 8,  8,  3,  3,  3,  3,  4,  4, 4,
                         4, 9, 9, 9, 9,  10, 10, 10, 10, 7,  7,  7, 7,
                         8, 8, 8, 8, 11, 11, 11, 11, 12, 12, 12, 12}};

    ref_interior_agglomerates = {{1, 3, 8, 9},
                                 {4, 5, 6, 7, 12, 13},
                                 {10, 11, 33, 34, 35},
                                 {14, 15, 36, 37, 38, 39},
                                 {16, 17, 18, 19, 24, 25},
                                 {20, 22, 28, 29},
                                 {26, 27, 48, 49, 50, 51},
                                 {30, 31, 52, 54, 55},
                                 {40, 41, 43},
                                 {44, 45, 46, 47},
                                 {56, 57, 58, 59},
                                 {60, 61, 62}};
    ref_halo_agglomerates = {
        {4, 6, 10, 11, 12, 14},
        {1, 3, 9, 11, 14, 15, 16, 18, 24, 26},
        {8, 9, 12, 14, 36, 38, 40, 41, 44},
        {9, 11, 12, 13, 24, 26, 33, 35, 41, 44, 45, 48, 50, 56},
        {5, 7, 13, 15, 20, 22, 26, 27, 28, 30},
        {17, 19, 25, 27, 30, 31},
        {13, 15, 24, 25, 28, 30, 37, 39, 45, 52, 54, 56, 57, 60},
        {25, 27, 28, 29, 49, 51, 57, 60, 61},
        {34, 35, 38, 44, 46},
        {35, 38, 39, 41, 43, 50, 56, 58},
        {39, 45, 47, 50, 51, 54, 60, 62},
        {51, 54, 55, 57, 59}};
  }
  else if (world_size == 2)
  {
    if (world_rank == 0)
    {
      ref_agglomerates = {{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2,
                           4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 7, 7, 6, 6, 8, 8,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
      ref_interior_agglomerates = {
          {5, 7, 12, 13},           {8, 9, 10, 11, 16, 17}, {14, 15}, {18, 19},
          {20, 21, 22, 23, 28, 29}, {24, 26, 32, 33},       {30, 31}, {34, 35}};
      ref_halo_agglomerates = {{8, 10, 14, 15, 16, 18},
                               {5, 7, 13, 15, 18, 19, 20, 22, 28, 30},
                               {12, 13, 16, 18, 36, 37, 40},
                               {13, 15, 16, 17, 28, 30, 37, 40, 41, 44},
                               {9, 11, 17, 19, 24, 26, 30, 31, 32, 34},
                               {21, 23, 29, 31, 34, 35},
                               {17, 19, 28, 29, 32, 34, 41, 44, 45, 48},
                               {29, 31, 32, 33, 45, 48, 49}};
    }
    else
    {
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 4, 4,
                           5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 7, 7, 6, 6, 8, 8}};
      ref_interior_agglomerates = {{20, 21, 23, 28, 29},
                                   {24, 25, 26, 27, 32, 33},
                                   {30, 31},
                                   {34, 35},
                                   {36, 37, 38, 39, 44, 45},
                                   {40, 41, 42, 48, 49},
                                   {46, 47},
                                   {50, 51}};
      ref_halo_agglomerates = {
          {6, 7, 10, 24, 26, 30, 31, 32, 34},
          {7, 10, 11, 14, 21, 23, 29, 31, 34, 35, 36, 38, 44, 46},
          {28, 29, 32, 34},
          {29, 31, 32, 33, 44, 46},
          {11, 14, 15, 18, 25, 27, 33, 35, 40, 42, 46, 47, 48, 50},
          {15, 18, 19, 37, 39, 45, 47, 50, 51},
          {33, 35, 44, 45, 48, 50},
          {45, 47, 48, 49}};
    }
  }
  else if (world_size == 4)
  {
    if (world_rank == 0)
    {
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                           1, 1, 3, 3, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
      ref_interior_agglomerates = {
          {8, 10, 15, 16}, {11, 12, 13, 14, 19, 20}, {17, 18}, {21, 22}};
      ref_halo_agglomerates = {{11, 13, 17, 18, 19, 21},
                               {8, 10, 16, 18, 21, 22, 23, 25, 27, 29},
                               {15, 16, 19, 21, 31, 32, 35},
                               {16, 18, 19, 20, 27, 29, 32, 35, 36, 39}};
    }
    if (world_rank == 1)
    {
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 4,
                           4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
      ref_interior_agglomerates = {
          {15, 16, 17, 18, 23, 24}, {19, 21, 27, 28}, {25, 26}, {29, 30}};
      ref_halo_agglomerates = {{8, 10, 12, 14, 19, 21, 25, 26, 27, 29},
                               {16, 18, 24, 26, 29, 30},
                               {12, 14, 23, 24, 27, 29, 32, 35, 36, 39},
                               {24, 26, 27, 28, 36, 39, 40}};
    }
    if (world_rank == 2)
    {
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 3,
                           3, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0}};
      ref_interior_agglomerates = {
          {19, 20, 22, 27, 28}, {23, 24, 25, 26, 31, 32}, {29, 30}, {33, 34}};
      ref_halo_agglomerates = {
          {9, 10, 13, 23, 25, 29, 30, 31, 33},
          {10, 13, 14, 17, 20, 22, 28, 30, 33, 34, 35, 37, 39, 41},
          {27, 28, 31, 33},
          {28, 30, 31, 32, 39, 41}};
    }
    if (world_rank == 3)
    {
      ref_agglomerates = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                           1, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 4, 4}};

      ref_interior_agglomerates = {
          {27, 28, 29, 30, 35, 36}, {31, 32, 33, 39, 40}, {37, 38}, {41, 42}};
      ref_halo_agglomerates = {
          {10, 13, 14, 17, 20, 22, 24, 26, 31, 33, 37, 38, 39, 41},
          {14, 17, 18, 28, 30, 36, 38, 41, 42},
          {24, 26, 35, 36, 39, 41},
          {36, 38, 39, 40}};
    }
  }

  BOOST_TEST(agglomerates == ref_agglomerates);
  for (unsigned int i = 0; i < ref_interior_agglomerates.size(); ++i)
    BOOST_TEST(boundary_agglomerates.first[i] == ref_interior_agglomerates[i]);
  for (unsigned int i = 0; i < ref_halo_agglomerates.size(); ++i)
    BOOST_TEST(boundary_agglomerates.second[i] == ref_halo_agglomerates[i]);
}
