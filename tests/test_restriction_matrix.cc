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

#define BOOST_TEST_MODULE restriction

#include "main.cc"

#include <mfmg/amge_host.hpp>

#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <random>

BOOST_AUTO_TEST_CASE(restriction_matrix)
{
  dealii::parallel::distributed::Triangulation<3> triangulation(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);
  dealii::FE_Q<3> fe(4);
  dealii::DoFHandler<3> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  mfmg::AMGe_host<3, dealii::LinearAlgebra::distributed::Vector<double>> amge(
      MPI_COMM_WORLD, dof_handler);

  unsigned int const rank =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  unsigned int const n_local_rows = rank + 3;
  unsigned int const eigenvectors_size = 10;
  std::vector<dealii::Vector<double>> eigenvectors(
      n_local_rows, dealii::Vector<double>(eigenvectors_size));
  for (unsigned int i = 0; i < n_local_rows; ++i)
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
      eigenvectors[i][j] =
          n_local_rows * eigenvectors_size + i * eigenvectors_size + j;

  std::vector<std::vector<dealii::types::global_dof_index>> dof_indices_maps(
      n_local_rows,
      std::vector<dealii::types::global_dof_index>(eigenvectors_size));
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, dof_handler.n_dofs() - 1);
  for (unsigned int i = 0; i < n_local_rows; ++i)
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
      dof_indices_maps[i][j] = distribution(generator);

  dealii::TrilinosWrappers::SparseMatrix restriction_sparse_matrix;
  amge.compute_restriction_sparse_matrix(eigenvectors, dof_indices_maps,
                                         restriction_sparse_matrix);

  // Check that the matrix was built correctly
  unsigned int row_offset = 0;
  for (unsigned int i = 0; i < rank; ++i)
    row_offset += i + 3;
  for (unsigned int i = 0; i < n_local_rows; ++i)
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
      BOOST_TEST(
          restriction_sparse_matrix(row_offset + i, dof_indices_maps[i][j]) ==
          eigenvectors[i][j]);
}
