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

#define BOOST_TEST_MODULE eigenvectors

#include "main.cc"

#include <mfmg/amge_host.hpp>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/trilinos_vector.h>

#include <algorithm>

namespace tt = boost::test_tools;
namespace ut = boost::unit_test;

void diagonal_matrices(dealii::DoFHandler<2> &dof_handler,
                       dealii::ConstraintMatrix &constraints,
                       dealii::SparsityPattern &system_sparsity_pattern,
                       dealii::SparseMatrix<double> &system_matrix,
                       dealii::SparsityPattern &mass_sparsity_pattern,
                       dealii::SparseMatrix<double> &mass_matrix)
{
  dealii::FE_Q<2> fe(1);
  dof_handler.distribute_dofs(fe);

  constraints.clear();

  unsigned int const size = 30;
  std::vector<std::vector<unsigned int>> column_indices(
      size, std::vector<unsigned int>(1));
  for (unsigned int i = 0; i < size; ++i)
    column_indices[i][0] = i;
  mass_sparsity_pattern.copy_from(size, size, column_indices.begin(),
                                  column_indices.end());
  system_sparsity_pattern.copy_from(mass_sparsity_pattern);
  system_matrix.reinit(system_sparsity_pattern);
  mass_matrix.reinit(mass_sparsity_pattern);
  for (unsigned int i = 0; i < size; ++i)
  {
    system_matrix.diag_element(i) = static_cast<double>(i + 1);
    mass_matrix.diag_element(i) = 1.;
  }
}

BOOST_AUTO_TEST_CASE(diagonal, *ut::tolerance(1e-12))
{
  dealii::parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(3);
  dealii::FE_Q<2> fe(1);
  dealii::DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  mfmg::AMGe_host<2, dealii::TrilinosWrappers::MPI::Vector> amge(MPI_COMM_WORLD,
                                                                 dof_handler);

  unsigned int const n_eigenvectors = 5;
  std::map<typename dealii::Triangulation<2>::active_cell_iterator,
           typename dealii::DoFHandler<2>::active_cell_iterator>
      patch_to_global_map;
  for (auto cell : dof_handler.active_cell_iterators())
    patch_to_global_map[cell] = cell;

  std::vector<std::complex<double>> eigenvalues;
  std::vector<dealii::Vector<double>> eigenvectors;
  std::vector<dealii::types::global_dof_index> dof_indices_map;
  std::tie(eigenvalues, eigenvectors, dof_indices_map) =
      amge.compute_local_eigenvectors(n_eigenvectors, 1e-13, triangulation,
                                      patch_to_global_map, diagonal_matrices);

  std::vector<dealii::types::global_dof_index> ref_dof_indices_map(
      dof_handler.n_dofs());
  std::iota(ref_dof_indices_map.begin(), ref_dof_indices_map.end(), 0);
  BOOST_TEST(dof_indices_map == ref_dof_indices_map, tt::per_element());

  unsigned int const eigenvector_size = eigenvectors[0].size();
  std::vector<std::complex<double>> ref_eigenvalues(n_eigenvectors);
  std::vector<dealii::Vector<double>> ref_eigenvectors(
      n_eigenvectors, dealii::Vector<double>(eigenvector_size));
  for (unsigned int i = 0; i < n_eigenvectors; ++i)
  {
    ref_eigenvalues[i] = static_cast<double>(i + 1);
    ref_eigenvectors[i][i] = 1.;
  }

  for (unsigned int i = 0; i < n_eigenvectors; ++i)
  {
    BOOST_TEST(eigenvalues[i].real() == ref_eigenvalues[i].real());
    BOOST_TEST(eigenvalues[i].imag() == ref_eigenvalues[i].imag());
    for (unsigned int j = 0; j < eigenvector_size; ++j)
      BOOST_TEST(std::abs(eigenvectors[i][j]) == ref_eigenvectors[i][j]);
  }
}

void constrained_diagonal_matrices(
    dealii::DoFHandler<2> &dof_handler, dealii::ConstraintMatrix &constraints,
    dealii::SparsityPattern &system_sparsity_pattern,
    dealii::SparseMatrix<double> &system_matrix,
    dealii::SparsityPattern &mass_sparsity_pattern,
    dealii::SparseMatrix<double> &mass_matrix)
{
  dealii::FE_Q<2> fe(1);
  dof_handler.distribute_dofs(fe);

  constraints.clear();
  constraints.add_line(0);
  constraints.close();

  unsigned int const size = 30;
  std::vector<std::vector<unsigned int>> column_indices(
      size, std::vector<unsigned int>(1));
  for (unsigned int i = 0; i < size; ++i)
    column_indices[i][0] = i;
  mass_sparsity_pattern.copy_from(size, size, column_indices.begin(),
                                  column_indices.end());
  system_sparsity_pattern.copy_from(mass_sparsity_pattern);
  system_matrix.reinit(system_sparsity_pattern);
  mass_matrix.reinit(mass_sparsity_pattern);
  for (unsigned int i = 0; i < size; ++i)
  {
    system_matrix.diag_element(i) = static_cast<double>(i + 1);
    mass_matrix.diag_element(i) = 1.;
  }
}

BOOST_AUTO_TEST_CASE(diagonal_constraint, *ut::tolerance(1e-12))
{
  dealii::parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(3);
  dealii::FE_Q<2> fe(1);
  dealii::DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  mfmg::AMGe_host<2, dealii::TrilinosWrappers::MPI::Vector> amge(MPI_COMM_WORLD,
                                                                 dof_handler);

  unsigned int const n_eigenvectors = 5;
  std::map<typename dealii::Triangulation<2>::active_cell_iterator,
           typename dealii::DoFHandler<2>::active_cell_iterator>
      patch_to_global_map;
  for (auto cell : dof_handler.active_cell_iterators())
    patch_to_global_map[cell] = cell;

  std::vector<std::complex<double>> eigenvalues;
  std::vector<dealii::Vector<double>> eigenvectors;
  std::vector<dealii::types::global_dof_index> dof_indices_map;
  std::tie(eigenvalues, eigenvectors, dof_indices_map) =
      amge.compute_local_eigenvectors(n_eigenvectors, 1e-13, triangulation,
                                      patch_to_global_map,
                                      constrained_diagonal_matrices);

  std::vector<dealii::types::global_dof_index> ref_dof_indices_map(
      dof_handler.n_dofs());
  std::iota(ref_dof_indices_map.begin(), ref_dof_indices_map.end(), 0);
  BOOST_TEST(dof_indices_map == ref_dof_indices_map, tt::per_element());

  unsigned int const eigenvector_size = eigenvectors[0].size();
  std::vector<std::complex<double>> ref_eigenvalues(n_eigenvectors);
  std::vector<dealii::Vector<double>> ref_eigenvectors(
      n_eigenvectors, dealii::Vector<double>(eigenvector_size));
  for (unsigned int i = 0; i < n_eigenvectors; ++i)
  {
    ref_eigenvalues[i] = static_cast<double>(i + 2);
    ref_eigenvectors[i][i + 1] = 1.;
  }

  for (unsigned int i = 0; i < n_eigenvectors; ++i)
  {
    BOOST_TEST(eigenvalues[i].real() == ref_eigenvalues[i].real());
    BOOST_TEST(eigenvalues[i].imag() == ref_eigenvalues[i].imag());
    for (unsigned int j = 0; j < eigenvector_size; ++j)
      BOOST_TEST(std::abs(eigenvectors[i][j]) == ref_eigenvectors[i][j]);
  }
}
