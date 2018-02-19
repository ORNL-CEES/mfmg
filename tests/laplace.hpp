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

#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/constraint_matrix.templates.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <string>

template <int dim, typename VectorType>
class Laplace
{
public:
  Laplace(MPI_Comm const &comm, unsigned int fe_degree);

  void setup_system();

  void assemble_system(dealii::Function<dim> const &source);

  template <typename PreconditionerType>
  VectorType solve(PreconditionerType &preconditioner);

  double compute_error(dealii::Function<dim> const &exact_solution);

  void output_results() const;

  template <typename PreconditionerType>
  void run(PreconditionerType &preconditioner,
           dealii::Function<dim> const &source);

private:
  MPI_Comm _comm;
  dealii::parallel::distributed::Triangulation<dim> _triangulation;
  dealii::FE_Q<dim> _fe;
  dealii::DoFHandler<dim> _dof_handler;
  dealii::IndexSet _locally_owned_dofs;
  dealii::IndexSet _locally_relevant_dofs;
  dealii::ConstraintMatrix _constraints;
  dealii::TrilinosWrappers::SparseMatrix _system_matrix;
  VectorType _locally_relevant_solution;
  VectorType _system_rhs;
};

template <int dim, typename VectorType>
Laplace<dim, VectorType>::Laplace(MPI_Comm const &comm, unsigned int fe_degree)
    : _comm(comm), _triangulation(_comm), _fe(fe_degree),
      _dof_handler(_triangulation)
{
}

template <int dim, typename VectorType>
void Laplace<dim, VectorType>::setup_system()
{
  dealii::GridGenerator::hyper_cube(_triangulation);
  _triangulation.refine_global(3);

  _dof_handler.distribute_dofs(_fe);

  // Get the IndexSets
  _locally_owned_dofs = _dof_handler.locally_owned_dofs();
  dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler,
                                                  _locally_relevant_dofs);

  // Resize the vectors
  _locally_relevant_solution.reinit(_locally_owned_dofs, _locally_relevant_dofs,
                                    _comm);
  _system_rhs.reinit(_locally_owned_dofs, _comm);

  // Compute the constraints
  _constraints.clear();
  _constraints.reinit(_locally_relevant_dofs);
  dealii::DoFTools::make_hanging_node_constraints(_dof_handler, _constraints);
  dealii::VectorTools::interpolate_boundary_values(
      _dof_handler, 0, dealii::Functions::ZeroFunction<dim>(), _constraints);
  _constraints.close();

  // Build the sparsity pattern
  dealii::TrilinosWrappers::SparsityPattern sparsity_pattern(
      _locally_owned_dofs, _comm);
  dealii::DoFTools::make_sparsity_pattern(_dof_handler, sparsity_pattern,
                                          _constraints);
  sparsity_pattern.compress();

  // Reinitialize the sparse matrix with the sparsity pattern
  _system_matrix.reinit(sparsity_pattern);
}

template <int dim, typename VectorType>
void Laplace<dim, VectorType>::assemble_system(
    dealii::Function<dim> const &source)
{
  unsigned int const fe_degree = _fe.degree;
  dealii::QGauss<dim> const quadrature(fe_degree + 1);
  dealii::FEValues<dim> fe_values(
      _fe, quadrature,
      dealii::update_values | dealii::update_gradients |
          dealii::update_quadrature_points | dealii::update_JxW_values);
  unsigned int const dofs_per_cell = _fe.dofs_per_cell;
  unsigned int const n_q_points = quadrature.size();
  dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  dealii::Vector<double> cell_rhs(dofs_per_cell);

  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (auto cell :
       dealii::filter_iterators(_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    cell_matrix = 0;
    cell_rhs = 0;
    fe_values.reinit(cell);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
      double const rhs_value =
          source.value(fe_values.quadrature_point(q_point));
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          cell_matrix(i, j) += fe_values.shape_grad(i, q_point) *
                               fe_values.shape_grad(j, q_point) *
                               fe_values.JxW(q_point);
        cell_rhs(i) += rhs_value * fe_values.shape_value(i, q_point) *
                       fe_values.JxW(q_point);
      }
    }

    cell->get_dof_indices(local_dof_indices);
    _constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, _system_matrix, _system_rhs);
  }

  _system_matrix.compress(dealii::VectorOperation::add);
  _system_rhs.compress(dealii::VectorOperation::add);
}

template <int dim, typename VectorType>
template <typename PreconditionerType>
VectorType Laplace<dim, VectorType>::solve(PreconditionerType &preconditioner)
{
  VectorType solution(_locally_owned_dofs, _comm);
  dealii::SolverControl solver_control(_dof_handler.n_dofs(),
                                       1e-12 * _system_rhs.l2_norm());
  dealii::TrilinosWrappers::SolverCG solver(solver_control);
  preconditioner.initialize(_system_matrix);
  solver.solve(_system_matrix, solution, _system_rhs, preconditioner);

  if (dealii::Utilities::MPI::this_mpi_process(_comm) == 0)
    std::cout << "Solved in " << solver_control.last_step() << " iterations."
              << std::endl;

  _constraints.distribute(solution);
  _locally_relevant_solution = solution;

  return solution;
}

template <int dim, typename VectorType>
double Laplace<dim, VectorType>::compute_error(
    dealii::Function<dim> const &exact_solution)
{
  unsigned int const fe_degree = _fe.degree;
  dealii::QGauss<dim> const quadrature(fe_degree + 1);

  dealii::Vector<double> difference(
      _dof_handler.get_triangulation().n_active_cells());
  dealii::VectorTools::integrate_difference(
      _dof_handler, _locally_relevant_solution, exact_solution, difference,
      quadrature, dealii::VectorTools::L2_norm);

  return dealii::VectorTools::compute_global_error(
      _dof_handler.get_triangulation(), difference,
      dealii::VectorTools::L2_norm);
}

template <int dim, typename VectorType>
void Laplace<dim, VectorType>::output_results() const
{
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(_dof_handler);
  data_out.add_data_vector(_locally_relevant_solution, "u");

  dealii::Vector<float> subdomain(_triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = _triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  std::string filename =
      "solution-" + std::to_string(_triangulation.locally_owned_subdomain());
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (dealii::Utilities::MPI::this_mpi_process(_comm) == 0)
  {
    std::vector<std::string> filenames;
    unsigned int comm_size = dealii::Utilities::MPI::n_mpi_processes(_comm);
    for (unsigned int i = 0; i < comm_size; ++i)
      filenames.push_back("solution-" + std::to_string(i) + ".vtu");
    std::ofstream master_output("solution.pvtu");
    data_out.write_pvtu_record(master_output, filenames);
  }
}

template <int dim, typename VectorType>
template <typename PreconditionerType>
void Laplace<dim, VectorType>::run(PreconditionerType &preconditioner,
                                   dealii::Function<dim> const &source)
{
  setup_system();
  assemble_system(source);
  solve(preconditioner);
  output_results();
}
