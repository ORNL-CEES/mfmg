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

#define BOOST_TEST_MODULE hierarchy

#include "main.cc"

#include "laplace.hpp"

#include <mfmg/adapters_dealii.hpp>
#include <mfmg/hierarchy.hpp>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_linear_operator.h>
#include <deal.II/lac/trilinos_vector.h>

#include <random>

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
  return 1.;
}

template <int dim, typename VectorType>
class TestMeshEvaluator : public mfmg::DealIIMeshEvaluator<dim, VectorType>
{
private:
  using value_type =
      typename mfmg::DealIIMeshEvaluator<dim, VectorType>::value_type;

protected:
  virtual void evaluate(dealii::DoFHandler<dim> &, dealii::ConstraintMatrix &,
                        dealii::TrilinosWrappers::SparsityPattern &,
                        dealii::TrilinosWrappers::SparseMatrix &system_matrix)
      const override final
  {
    // TODO this is pretty expansive, we should use a shared pointer
    system_matrix.copy_from(_matrix);
  }

  virtual void
  evaluate(dealii::DoFHandler<dim> &dof_handler,
           dealii::ConstraintMatrix &constraints,
           dealii::SparsityPattern &system_sparsity_pattern,
           dealii::SparseMatrix<value_type> &system_matrix) const override final
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

public:
  TestMeshEvaluator(const dealii::TrilinosWrappers::SparseMatrix &matrix)
      : _matrix(matrix)
  {
  }

private:
  const dealii::TrilinosWrappers::SparseMatrix &_matrix;
};

BOOST_AUTO_TEST_CASE(hierarchy_2d)
{
  unsigned int constexpr dim = 2;
  using DVector = dealii::TrilinosWrappers::MPI::Vector;
  using MeshEvaluator = mfmg::DealIIMeshEvaluator<dim, DVector>;
  using Mesh = mfmg::DealIIMesh<dim>;

  MPI_Comm comm = MPI_COMM_WORLD;

  dealii::ConditionalOStream pcout(
      std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  Source<dim> source;

  const int num_refinements = 5;

  Laplace<dim, DVector> laplace(comm, 1);
  laplace.setup_system();
  laplace.assemble_system(source);

  auto mesh =
      std::make_shared<Mesh>(laplace._dof_handler, laplace._constraints);

  auto const &a = laplace._system_matrix;
  auto const locally_owned_dofs = laplace._locally_owned_dofs;
  DVector solution(locally_owned_dofs, comm);
  DVector rhs(laplace._system_rhs);

  const int local_size = rhs.local_size();

  std::default_random_engine generator;
  std::uniform_real_distribution<typename DVector::value_type> distribution(0.,
                                                                            1.);
  for (auto const index : locally_owned_dofs)
    solution[index] = distribution(generator);

  auto params = std::make_shared<boost::property_tree::ptree>();
  params->put("eigensolver: number of eigenvectors", 1);
  params->put("eigensolver: tolerance", 1e-14);
  params->put("agglomeration: nx", 2);
  params->put("agglomeration: ny", 2);

  TestMeshEvaluator<dim, DVector> evaluator(a);
  mfmg::Hierarchy<MeshEvaluator, DVector> hierarchy(comm, evaluator, *mesh,
                                                    params);

  // We want to do 20 V-cycle iterations. The rhs of is zero.
  DVector residual(rhs);
  unsigned int const n_cycles = 20;
  std::cout << std::scientific;
  for (unsigned int i = 0; i < n_cycles; ++i)
  {
    laplace._system_matrix.vmult(residual, solution);
    pcout << "Residual before: " << residual.l2_norm() << std::endl;
    hierarchy.apply(rhs, solution);
    laplace._system_matrix.vmult(residual, solution);
    pcout << "Residual after: " << residual.l2_norm() << std::endl;
  }
}
