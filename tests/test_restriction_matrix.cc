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

#include <mfmg/adapters_dealii.hpp>
#include <mfmg/amge_host.hpp>

#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <random>

BOOST_AUTO_TEST_CASE(restriction_matrix)
{
  unsigned int constexpr dim = 2;
  using Vector = dealii::LinearAlgebra::distributed::Vector<double>;
  using DummyMeshEvaluator = mfmg::DealIIMeshEvaluator<dim, Vector>;

  MPI_Comm comm = MPI_COMM_WORLD;
  dealii::parallel::distributed::Triangulation<dim> triangulation(comm);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);
  dealii::FE_Q<dim> fe(1);
  dealii::DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  DummyMeshEvaluator evaluator;
  mfmg::AMGe_host<dim, DummyMeshEvaluator,
                  dealii::LinearAlgebra::distributed::Vector<double>>
      amge(comm, dof_handler);

  auto const locally_owned_dofs = dof_handler.locally_owned_dofs();
  unsigned int const n_local_rows = locally_owned_dofs.n_elements();

  // Fill the eigenvectors
  unsigned int const eigenvectors_size = 3;
  std::vector<dealii::Vector<double>> eigenvectors(
      n_local_rows, dealii::Vector<double>(eigenvectors_size));
  for (unsigned int i = 0; i < n_local_rows; ++i)
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
      eigenvectors[i][j] =
          n_local_rows * eigenvectors_size + i * eigenvectors_size + j;

  // Fill dof_indices_maps
  std::vector<std::vector<dealii::types::global_dof_index>> dof_indices_maps(
      n_local_rows,
      std::vector<dealii::types::global_dof_index>(eigenvectors_size));
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, dof_handler.n_dofs() - 1);
  for (unsigned int i = 0; i < n_local_rows; ++i)
  {
    // We don't want dof_indices to have repeated values in a row
    std::set<int> dofs_set;
    unsigned int j = 0;
    while (dofs_set.size() < eigenvectors_size)
    {
      int dof_index = distribution(generator);
      if ((dofs_set.count(dof_index) == 0) &&
          (locally_owned_dofs.is_element(dof_index)))
      {
        dof_indices_maps[i][j] = dof_index;
        dofs_set.insert(dof_index);
        ++j;
      }
    }
  }

  // Fill diag_elements
  std::vector<std::vector<double>> diag_elements(
      n_local_rows, std::vector<double>(eigenvectors_size, 1));

  // Fill n_local_eigenvectors
  std::vector<unsigned int> n_local_eigenvectors(n_local_rows, 1);

  // Fill system_sparse_matrix
  dealii::TrilinosWrappers::SparseMatrix system_sparse_matrix(
      locally_owned_dofs, comm);
  for (auto const index : locally_owned_dofs)
    system_sparse_matrix.set(index, index, 1.0);
  system_sparse_matrix.compress(dealii::VectorOperation::insert);

  dealii::TrilinosWrappers::SparseMatrix restriction_sparse_matrix;
  amge.compute_restriction_sparse_matrix(
      eigenvectors, diag_elements, dof_indices_maps, n_local_eigenvectors,
      system_sparse_matrix, restriction_sparse_matrix);

  // Check that the matrix was built correctly
  auto restriction_locally_owned_dofs =
      restriction_sparse_matrix.locally_owned_range_indices();
  unsigned int pos = 0;
  for (auto const index : restriction_locally_owned_dofs)
  {
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
      BOOST_TEST(restriction_sparse_matrix(index, dof_indices_maps[pos][j]) ==
                 eigenvectors[pos][j]);
    ++pos;
  }
}
