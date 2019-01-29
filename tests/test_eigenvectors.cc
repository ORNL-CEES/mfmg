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

#define BOOST_TEST_MODULE eigenvectors

#include <mfmg/dealii/amge_host.hpp>
#include <mfmg/dealii/dealii_mesh_evaluator.hpp>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>

#include <algorithm>

#include "main.cc"

namespace tt = boost::test_tools;
namespace ut = boost::unit_test;

template <int dim>
class DiagonalTestMeshEvaluator : public mfmg::DealIIMeshEvaluator<dim>
{
public:
  DiagonalTestMeshEvaluator(dealii::DoFHandler<dim> &dof_handler,
                            dealii::AffineConstraints<double> &constraints)
      : mfmg::DealIIMeshEvaluator<dim>(dof_handler, constraints)
  {
  }

  // Diagonal matrices. We only need local evaluate function.
  void evaluate_agglomerate(dealii::DoFHandler<dim> &dof_handler,
                            dealii::AffineConstraints<double> &constraints,
                            dealii::TrilinosWrappers::SparseMatrix
                                &system_matrix) const override final
  {
    dealii::FE_Q<2> fe(1);
    dof_handler.distribute_dofs(fe);

    constraints.clear();

    unsigned int const n_dofs = dof_handler.n_dofs();
    dealii::DynamicSparsityPattern system_sparsity_pattern(n_dofs);
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      system_sparsity_pattern.add(i, i);
    }
    system_sparsity_pattern.compress();
    system_matrix.reinit(dealii::complete_index_set(n_dofs),
                         system_sparsity_pattern, MPI_COMM_SELF);
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      system_matrix.set(i, i, static_cast<double>(i + 1));
    }
    system_matrix.compress(dealii::VectorOperation::insert);
  }

  void
  evaluate_global(dealii::DoFHandler<dim> &,
                  dealii::AffineConstraints<double> &,
                  dealii::TrilinosWrappers::SparseMatrix &) const override final
  {
  }
};

BOOST_AUTO_TEST_CASE(diagonal, *ut::tolerance(1e-12))
{
  const int dim = 2;
  using Vector = dealii::LinearAlgebra::distributed::Vector<double>;
  using MeshEvaluator = mfmg::DealIIMeshEvaluator<2>;

  dealii::parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(3);
  dealii::FE_Q<2> fe(1);
  dealii::DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  mfmg::AMGe_host<2, MeshEvaluator, Vector> amge(MPI_COMM_WORLD, dof_handler);

  unsigned int const n_eigenvectors = 5;
  std::map<typename dealii::Triangulation<2>::active_cell_iterator,
           typename dealii::DoFHandler<2>::active_cell_iterator>
      patch_to_global_map;
  for (auto cell : dof_handler.active_cell_iterators())
    patch_to_global_map[cell] = cell;

  dealii::AffineConstraints<double> constraints;
  DiagonalTestMeshEvaluator<dim> evaluator(dof_handler, constraints);
  std::vector<std::complex<double>> eigenvalues;
  std::vector<dealii::Vector<double>> eigenvectors;
  std::vector<double> diag_elements;
  std::vector<dealii::types::global_dof_index> dof_indices_map;
  std::tie(eigenvalues, eigenvectors, diag_elements, dof_indices_map) =
      amge.compute_local_eigenvectors(n_eigenvectors, 1e-13, triangulation,
                                      patch_to_global_map, evaluator);

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

template <int dim>
class ConstrainedDiagonalTestMeshEvaluator
    : public mfmg::DealIIMeshEvaluator<dim>
{
public:
  ConstrainedDiagonalTestMeshEvaluator(
      dealii::DoFHandler<dim> &dof_handler,
      dealii::AffineConstraints<double> &constraints)
      : mfmg::DealIIMeshEvaluator<dim>(dof_handler, constraints)
  {
  }

  // Diagonal matrices. We only need local evaluate function.
  void evaluate_agglomerate(dealii::DoFHandler<dim> &dof_handler,
                            dealii::AffineConstraints<double> &constraints,
                            dealii::TrilinosWrappers::SparseMatrix
                                &system_matrix) const override final
  {
    dealii::FE_Q<2> fe(1);
    dof_handler.distribute_dofs(fe);

    constraints.clear();
    constraints.add_line(0);
    constraints.close();

    unsigned int const n_dofs = dof_handler.n_dofs();
    dealii::DynamicSparsityPattern system_sparsity_pattern(n_dofs);
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      system_sparsity_pattern.add(i, i);
    }
    system_sparsity_pattern.compress();
    system_matrix.reinit(dealii::complete_index_set(n_dofs),
                         system_sparsity_pattern, MPI_COMM_SELF);
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      system_matrix.set(i, i, static_cast<double>(i + 1));
    }
    system_matrix.compress(dealii::VectorOperation::insert);
  }

  void
  evaluate_global(dealii::DoFHandler<dim> &,
                  dealii::AffineConstraints<double> &,
                  dealii::TrilinosWrappers::SparseMatrix &) const override final
  {
  }
};

BOOST_AUTO_TEST_CASE(diagonal_constraint, *ut::tolerance(1e-12))
{
  const int dim = 2;
  using Vector = dealii::LinearAlgebra::distributed::Vector<double>;
  using MeshEvaluator = mfmg::DealIIMeshEvaluator<2>;

  dealii::parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(3);
  dealii::FE_Q<2> fe(1);
  dealii::DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  mfmg::AMGe_host<2, MeshEvaluator, Vector> amge(MPI_COMM_WORLD, dof_handler);

  unsigned int const n_eigenvectors = 5;
  std::map<typename dealii::Triangulation<2>::active_cell_iterator,
           typename dealii::DoFHandler<2>::active_cell_iterator>
      patch_to_global_map;
  for (auto cell : dof_handler.active_cell_iterators())
    patch_to_global_map[cell] = cell;

  dealii::AffineConstraints<double> constraints;
  ConstrainedDiagonalTestMeshEvaluator<dim> evaluator(dof_handler, constraints);
  std::vector<std::complex<double>> eigenvalues;
  std::vector<dealii::Vector<double>> eigenvectors;
  std::vector<double> diag_elements;
  std::vector<dealii::types::global_dof_index> dof_indices_map;
  std::tie(eigenvalues, eigenvectors, diag_elements, dof_indices_map) =
      amge.compute_local_eigenvectors(n_eigenvectors, 1e-13, triangulation,
                                      patch_to_global_map, evaluator);

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
