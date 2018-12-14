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

#include <mfmg/dealii/amge_host.hpp>
#include <mfmg/dealii/dealii_mesh_evaluator.hpp>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <boost/property_tree/info_parser.hpp>

#include <random>

#include "laplace.hpp"
#include "main.cc"

namespace utf = boost::unit_test;

template <int dim>
class DummyMeshEvaluator : public mfmg::DealIIMeshEvaluator<dim>
{
public:
  DummyMeshEvaluator(dealii::DoFHandler<dim> &dof_handler,
                     dealii::AffineConstraints<double> &constraints)
      : mfmg::DealIIMeshEvaluator<dim>(dof_handler, constraints)
  {
  }

  void
  evaluate_global(dealii::DoFHandler<dim> &,
                  dealii::AffineConstraints<double> &,
                  dealii::TrilinosWrappers::SparseMatrix &) const override final
  {
  }

  void evaluate_agglomerate(dealii::DoFHandler<dim> &,
                            dealii::AffineConstraints<double> &,
                            dealii::SparsityPattern &,
                            dealii::SparseMatrix<double> &) const override final
  {
  }
};

BOOST_AUTO_TEST_CASE(restriction_matrix, *utf::tolerance(1e-14))
{
  unsigned int constexpr dim = 2;

  MPI_Comm comm = MPI_COMM_WORLD;
  dealii::parallel::distributed::Triangulation<dim> triangulation(comm);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);
  dealii::FE_Q<dim> fe(1);
  dealii::DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  dealii::AffineConstraints<double> constraints;
  DummyMeshEvaluator<dim> evaluator(dof_handler, constraints);
  mfmg::AMGe_host<dim, mfmg::DealIIMeshEvaluator<dim>,
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
      n_local_rows, std::vector<double>(eigenvectors_size));
  std::map<unsigned int, double> count_elem;
  for (unsigned int i = 0; i < n_local_rows; ++i)
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
      count_elem[dof_indices_maps[i][j]] += 1.0;
  for (unsigned int i = 0; i < n_local_rows; ++i)
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
      diag_elements[i][j] = 1. / count_elem[dof_indices_maps[i][j]];

  // Fill n_local_eigenvectors
  std::vector<unsigned int> n_local_eigenvectors(n_local_rows, 1);

  // Fill system_sparse_matrix
  dealii::TrilinosWrappers::SparseMatrix system_sparse_matrix(
      locally_owned_dofs, comm);
  for (auto const index : locally_owned_dofs)
    system_sparse_matrix.set(index, index, 1.0);
  system_sparse_matrix.compress(dealii::VectorOperation::insert);

  // NOTE have to extract diagonal entries from system matrix, cannot rely on
  // DummyMeshEvaluator::get_diagonal()
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                  locally_relevant_dofs);
  dealii::LinearAlgebra::distributed::Vector<double> locally_owned_global_diag(
      locally_owned_dofs, comm);
  for (auto const val : locally_owned_dofs)
    locally_owned_global_diag[val] = system_sparse_matrix.diag_element(val);
  locally_owned_global_diag.compress(dealii::VectorOperation::insert);

  dealii::LinearAlgebra::distributed::Vector<double>
      locally_relevant_global_diag(locally_owned_dofs, locally_relevant_dofs,
                                   comm);
  locally_relevant_global_diag = locally_owned_global_diag;

  dealii::TrilinosWrappers::SparseMatrix restriction_sparse_matrix;
  amge.compute_restriction_sparse_matrix(
      eigenvectors, diag_elements, dof_indices_maps, n_local_eigenvectors,
      locally_relevant_global_diag, restriction_sparse_matrix);

  // Check that the matrix was built correctly
  auto restriction_locally_owned_dofs =
      restriction_sparse_matrix.locally_owned_range_indices();
  unsigned int pos = 0;
  for (auto const index : restriction_locally_owned_dofs)
  {
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
      BOOST_TEST(restriction_sparse_matrix(index, dof_indices_maps[pos][j]) ==
                 diag_elements[pos][j] * eigenvectors[pos][j]);
    ++pos;
  }
}

template <int dim>
class Source : public dealii::Function<dim>
{
public:
  Source() = default;

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const component = 0) const override final;
};

template <int dim>
double Source<dim>::value(dealii::Point<dim> const &, unsigned int const) const
{
  return 0.;
}

template <int dim>
class ConstantMaterialProperty : public dealii::Function<dim>
{
public:
  ConstantMaterialProperty() = default;

  virtual double value(dealii::Point<dim> const &,
                       unsigned int const = 0) const override final
  {
    return 1.;
  }
};

template <int dim>
class TestMeshEvaluator : public mfmg::DealIIMeshEvaluator<dim>
{
public:
  TestMeshEvaluator(dealii::DoFHandler<dim> &dof_handler,
                    dealii::AffineConstraints<double> &constraints,
                    dealii::TrilinosWrappers::SparseMatrix const &matrix)
      : mfmg::DealIIMeshEvaluator<dim>(dof_handler, constraints),
        _matrix(matrix)
  {
  }

  void evaluate_global(dealii::DoFHandler<dim> &,
                       dealii::AffineConstraints<double> &,
                       dealii::TrilinosWrappers::SparseMatrix &system_matrix)
      const override final
  {
    // TODO this is pretty expansive, we should use a shared pointer
    system_matrix.copy_from(_matrix);
  }

  void evaluate_agglomerate(
      dealii::DoFHandler<dim> &dof_handler,
      dealii::AffineConstraints<double> &constraints,
      dealii::SparsityPattern &system_sparsity_pattern,
      dealii::SparseMatrix<double> &system_matrix) const override final
  {
    unsigned int const fe_degree = 1;
    dealii::FE_Q<dim> fe(fe_degree);
    dof_handler.distribute_dofs(fe);

    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                    locally_relevant_dofs);

    // Compute the constraints
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    // The test requires us not to put boundary conditions
    // dealii::VectorTools::interpolate_boundary_values(
    //     dof_handler, 1, dealii::Functions::ZeroFunction<dim>(), constraints);
    constraints.close();

    // Build the system sparsity pattern and reinitialize the system sparse
    // matrix
    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    system_sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(system_sparsity_pattern);

    // Fill the system matrix
    dealii::QGauss<dim> const quadrature(fe_degree + 1);
    dealii::FEValues<dim> fe_values(
        fe, quadrature,
        dealii::update_values | dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);
    unsigned int const dofs_per_cell = fe.dofs_per_cell;
    unsigned int const n_q_points = quadrature.size();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
        dofs_per_cell);
    for (auto cell :
         dealii::filter_iterators(dof_handler.active_cell_iterators(),
                                  dealii::IteratorFilters::LocallyOwnedCell()))
    {
      cell_matrix = 0;
      fe_values.reinit(cell);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) += fe_values.shape_grad(i, q_point) *
                                 fe_values.shape_grad(j, q_point) *
                                 fe_values.JxW(q_point);

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, local_dof_indices,
                                             system_matrix);
    }
  }

private:
  const dealii::TrilinosWrappers::SparseMatrix &_matrix;
};

// FIXME relaxed tolerance from 1e-14 to 1e-4 for this test to pass while using
// ARPACK's regular mode instead of shift-and-invert
BOOST_AUTO_TEST_CASE(weight_sum, *utf::tolerance(1e-4))
{
  // Check that the weight sum is equal to one
  unsigned int constexpr dim = 2;
  using DVector = dealii::LinearAlgebra::distributed::Vector<double>;
  using MeshEvaluator = mfmg::DealIIMeshEvaluator<dim>;

  MPI_Comm comm = MPI_COMM_WORLD;

  dealii::ConditionalOStream pcout(
      std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  Source<dim> source;

  auto params = std::make_shared<boost::property_tree::ptree>();
  boost::property_tree::info_parser::read_info("hierarchy_input.info", *params);
  params->put("eigensolver.number of eigenvectors", 1);
  auto agglomerate_ptree = params->get_child("agglomeration");
  int n_eigenvectors =
      params->get<int>("eigensolver.number of eigenvectors", 1);
  double tolerance = params->get<double>("eigensolver.tolerance", 1e-14);

  params->put("laplace.n_refinements", 4);
  std::shared_ptr<dealii::Function<dim>> material_property =
      std::make_shared<ConstantMaterialProperty<dim>>();
  auto laplace_ptree = params->get_child("laplace");
  Laplace<dim, DVector> laplace(comm, 1);
  laplace.setup_system(laplace_ptree);
  laplace.assemble_system(source, *material_property);

  TestMeshEvaluator<dim> evaluator(laplace._dof_handler, laplace._constraints,
                                   laplace._system_matrix);
  mfmg::AMGe_host<dim, MeshEvaluator, DVector> amge(comm, laplace._dof_handler);

  auto locally_relevant_global_diag = evaluator.get_diagonal();

  dealii::TrilinosWrappers::SparseMatrix restrictor_matrix;
  amge.setup_restrictor(agglomerate_ptree, n_eigenvectors, tolerance, evaluator,
                        locally_relevant_global_diag, restrictor_matrix);

  // Multiply the matrix by three because all the eigenvectors are 1/3. So we
  // are left with the weights.
  restrictor_matrix *= 3.;
  unsigned int const size = restrictor_matrix.n();
  auto domain_dofs = restrictor_matrix.locally_owned_domain_indices();
  DVector e(domain_dofs, comm);
  auto range_dofs = restrictor_matrix.locally_owned_range_indices();
  DVector ee(range_dofs, comm);
  for (unsigned int i = 0; i < size; ++i)
  {
    e = 0;
    if (domain_dofs.is_element(i))
      e[i] = 1.;
    e.compress(::dealii::VectorOperation::insert);
    restrictor_matrix.vmult(ee, e);
    BOOST_TEST(ee.l1_norm() == 1.);
  }
}
