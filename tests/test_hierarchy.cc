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

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_linear_operator.h>
#include <deal.II/lac/trilinos_vector.h>

#include <random>

unsigned int constexpr dim = 2;
using ScalarType = double;
using Vector = typename dealii::LinearAlgebra::distributed::Vector<ScalarType>;

template <int dim>
class Source : public dealii::Function<dim>
{
public:
  Source() = default;

  double value(dealii::Point<dim> const &p,
               unsigned int const component = 0) const override;
};

template <int dim>
double Source<dim>::value(dealii::Point<dim> const &p, unsigned int const) const
{
  double val = 0.;
  for (unsigned int d = 0; d < dim; ++d)
  {
    double tmp = 0.;
    for (unsigned int i = 0; i < dim; ++i)
      tmp += p[i] * (1 + p[i] * (1 + p[i] * (1 + p[i] * (1 + p[i]))));

    val += 2. * tmp;
  }

  return val;
}

template <int dim, class Vector>
class TestMeshEvaluator : public mfmg::DealIIMeshEvaluator<dim, Vector>
{
private:
  using value_type =
      typename mfmg::DealIIMeshEvaluator<dim, Vector>::value_type;

protected:
  // diagonal matrices
  void
  evaluate(dealii::DoFHandler<dim> &dof_handler,
           dealii::ConstraintMatrix &constraints,
           dealii::TrilinosWrappers::SparsityPattern &system_sparsity_pattern,
           dealii::TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix.copy_from(_matrix);
  }
  void evaluate(dealii::DoFHandler<dim> &dof_handler,
                dealii::ConstraintMatrix &constraints,
                dealii::SparsityPattern &system_sparsity_pattern,
                dealii::SparseMatrix<value_type> &system_matrix) const
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
    // TODO what should we do with the boundary condition?
    // dealii::VectorTools::interpolate_boundary_values(
    //     _dof_handler, 0, dealii::Functions::ZeroFunction<dim>(),
    //     _constraints);
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
  using Vector = dealii::LinearAlgebra::distributed::Vector<double>;
  using MeshEvaluator = mfmg::DealIIMeshEvaluator<dim, Vector>;
  using Operator = typename MeshEvaluator::global_operator_type;
  using Mesh = mfmg::DealIIMesh<dim>;

  MPI_Comm comm = MPI_COMM_WORLD;

  Source<dim> source;

  const int num_refinements = 6;

  Laplace<dim, Vector> laplace(comm, 1);
  laplace.setup_system(num_refinements);
  laplace.assemble_system(source);

  auto mesh =
      std::make_shared<Mesh>(laplace._dof_handler, laplace._constraints);

  const auto &a = laplace._system_matrix;
  Vector solution(laplace._locally_owned_dofs, comm);
  Vector rhs(laplace._system_rhs);

  const int local_size = rhs.local_size();

  std::default_random_engine generator;
  std::uniform_real_distribution<typename Vector::value_type> distribution(0.,
                                                                           1.);
  for (int i = 0; i < local_size; i++)
    solution.local_element(i) = distribution(generator);
  a.vmult(rhs, solution);

  auto params = std::make_shared<boost::property_tree::ptree>();
  params->put("eigensolver: number of eigenvectors", 1);
  params->put("eigensolver: tolerance", 1e-14);
  params->put<unsigned int>("agglomeration: nx", 2);
  params->put<unsigned int>("agglomeration: ny", 2);
  params->put<unsigned int>("agglomeration: nz", 2);

  TestMeshEvaluator<dim, Vector> evaluator(a);
  mfmg::Hierarchy<MeshEvaluator, Vector> hierarchy(comm, evaluator, *mesh,
                                                   params);

  dealii::SolverControl solver_control(laplace._dof_handler.n_dofs(),
                                       1e-12 * rhs.l2_norm());
  dealii::SolverCG<Vector> solver(solver_control);

  solution = 0.;
  solver.solve(laplace._system_matrix, solution, rhs, hierarchy);

  if (dealii::Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "AMGe: Solved in " << solver_control.last_step()
              << " iterations." << std::endl;

  laplace._constraints.distribute(solution);
  laplace._locally_relevant_solution = solution;

  solution = 0.;
  solver.solve(laplace._system_matrix, solution, rhs,
               dealii::PreconditionIdentity());
  if (dealii::Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "Identity: Solved in " << solver_control.last_step()
              << " iterations." << std::endl;

  solution = 0.;
  dealii::TrilinosWrappers::PreconditionSSOR ssor;
  ssor.initialize(laplace._system_matrix);
  solver.solve(laplace._system_matrix, solution, rhs, ssor);
  if (dealii::Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "SSOR: Solved in " << solver_control.last_step()
              << " iterations." << std::endl;
}
