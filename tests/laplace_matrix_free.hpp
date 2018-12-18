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

#ifndef MFMG_LAPLACE_MATRIX_FREE_HPP
#define MFMG_LAPLACE_MATRIX_FREE_HPP

#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/property_tree/ptree.hpp>

template <int dim, int fe_degree, typename ScalarType>
class LaplaceOperator
    : public dealii::MatrixFreeOperators::Base<
          dim, dealii::LinearAlgebra::distributed::Vector<ScalarType>>
{
public:
  typedef ScalarType value_type;

  LaplaceOperator();

  virtual void compute_diagonal() override final;

  template <typename MaterialPropertyType>
  void evaluate_coefficient(MaterialPropertyType const &material_property);

  // private:
  virtual void
  apply_add(dealii::LinearAlgebra::distributed::Vector<ScalarType> &dst,
            dealii::LinearAlgebra::distributed::Vector<ScalarType> const &src)
      const override final;

  void
  local_apply(dealii::MatrixFree<dim, ScalarType> const &matrix_free_data,
              dealii::LinearAlgebra::distributed::Vector<ScalarType> &dst,
              dealii::LinearAlgebra::distributed::Vector<ScalarType> const &src,
              std::pair<unsigned int, unsigned int> const &cell_range) const;

  void local_compute_diagonal(
      dealii::MatrixFree<dim, ScalarType> const &matrix_free_data,
      dealii::LinearAlgebra::distributed::Vector<ScalarType> &dst,
      unsigned int const &dummy,
      std::pair<unsigned int, unsigned int> const &cell_range) const;

  dealii::Table<2, dealii::VectorizedArray<ScalarType>> _coefficient;
};

template <int dim, int fe_degree, typename ScalarType>
LaplaceOperator<dim, fe_degree, ScalarType>::LaplaceOperator()
    : dealii::MatrixFreeOperators::Base<
          dim, dealii::LinearAlgebra::distributed::Vector<ScalarType>>()
{
}

template <int dim, int fe_degree, typename ScalarType>
void LaplaceOperator<dim, fe_degree, ScalarType>::compute_diagonal()
{
  this->inverse_diagonal_entries.reset(
      new dealii::DiagonalMatrix<
          dealii::LinearAlgebra::distributed::Vector<ScalarType>>());
  dealii::LinearAlgebra::distributed::Vector<ScalarType> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
  this->data->initialize_dof_vector(inverse_diagonal);
  unsigned int const dummy = 0;
  this->data->cell_loop(&LaplaceOperator::local_compute_diagonal, this,
                        inverse_diagonal, dummy);

  this->set_constrained_entries_to_one(inverse_diagonal);

  for (unsigned int i = 0; i < inverse_diagonal.local_size(); ++i)
    inverse_diagonal.local_element(i) = 1. / inverse_diagonal.local_element(i);
}

template <int dim, int fe_degree, typename ScalarType>
template <typename MaterialPropertyType>
void LaplaceOperator<dim, fe_degree, ScalarType>::evaluate_coefficient(
    MaterialPropertyType const &material_property)
{
  int constexpr n_q_points = fe_degree + 1;
  int constexpr n_components = 1;
  unsigned int const n_cells = this->data->n_macro_cells();
  dealii::FEEvaluation<dim, fe_degree, n_q_points, n_components, ScalarType>
      fe_eval(*this->data);

  _coefficient.reinit(n_cells, fe_eval.n_q_points);
  for (unsigned int cell = 0; cell < n_cells; ++cell)
  {
    fe_eval.reinit(cell);
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      _coefficient(cell, q) =
          material_property.value(fe_eval.quadrature_point(q));
  }
}

template <int dim, int fe_degree, typename ScalarType>
void LaplaceOperator<dim, fe_degree, ScalarType>::apply_add(
    dealii::LinearAlgebra::distributed::Vector<ScalarType> &dst,
    dealii::LinearAlgebra::distributed::Vector<ScalarType> const &src) const
{
  this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src);
}

template <int dim, int fe_degree, typename ScalarType>
void LaplaceOperator<dim, fe_degree, ScalarType>::local_apply(
    dealii::MatrixFree<dim, ScalarType> const &matrix_free_data,
    dealii::LinearAlgebra::distributed::Vector<ScalarType> &dst,
    dealii::LinearAlgebra::distributed::Vector<ScalarType> const &src,
    std::pair<unsigned int, unsigned int> const &cell_range) const
{
  int constexpr n_q_points = fe_degree + 1;
  int constexpr n_components = 1;
  dealii::FEEvaluation<dim, fe_degree, n_q_points, n_components, ScalarType>
      fe_eval(matrix_free_data);

  bool const evaluate_values = false;
  bool const evaluate_gradients = true;
  bool const integrate_values = false;
  bool const integrate_gradients = true;
  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(evaluate_values, evaluate_gradients);
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      fe_eval.submit_gradient(_coefficient(cell, q) * fe_eval.get_gradient(q),
                              q);
    fe_eval.integrate(integrate_values, integrate_gradients);
    fe_eval.distribute_local_to_global(dst);
  }
}

template <int dim, int fe_degree, typename ScalarType>
void LaplaceOperator<dim, fe_degree, ScalarType>::local_compute_diagonal(
    dealii::MatrixFree<dim, ScalarType> const &matrix_free_data,
    dealii::LinearAlgebra::distributed::Vector<ScalarType> &dst,
    unsigned int const &,
    std::pair<unsigned int, unsigned int> const &cell_range) const
{
  int constexpr n_q_points = fe_degree + 1;
  int constexpr n_components = 1;
  dealii::FEEvaluation<dim, fe_degree, n_q_points, n_components, ScalarType>
      fe_eval(matrix_free_data);

  dealii::AlignedVector<dealii::VectorizedArray<ScalarType>> diagonal(
      fe_eval.dofs_per_cell);

  bool const evaluate_values = false;
  bool const evaluate_gradients = true;
  bool const integrate_values = false;
  bool const integrate_gradients = true;
  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval.reinit(cell);
    for (unsigned int i = 0; i < fe_eval.dofs_per_cell; ++i)
    {
      for (unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
        fe_eval.submit_dof_value(dealii::VectorizedArray<ScalarType>(), j);
      fe_eval.submit_dof_value(dealii::make_vectorized_array<ScalarType>(1.),
                               i);

      fe_eval.evaluate(evaluate_values, evaluate_gradients);
      for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        fe_eval.submit_gradient(_coefficient(cell, q) * fe_eval.get_gradient(q),
                                q);

      fe_eval.integrate(integrate_values, integrate_gradients);
      diagonal[i] = fe_eval.get_dof_value(i);
    }
    for (unsigned int i = 0; i < fe_eval.dofs_per_cell; ++i)
      fe_eval.submit_dof_value(diagonal[i], i);
    fe_eval.distribute_local_to_global(dst);
  }
}

template <int dim, int fe_degree, typename ScalarType>
class LaplaceMatrixFree
{
public:
  LaplaceMatrixFree(MPI_Comm const &comm);

  template <typename MaterialPropertyType>
  void setup_system(boost::property_tree::ptree const &ptree,
                    MaterialPropertyType const &material_property);

  template <typename SourceType>
  void assemble_rhs(SourceType const &source);

  template <typename PreconditionerType>
  void solve(PreconditionerType &preconditioner);

  double compute_error(dealii::Function<dim> const &exact_solution);

  // The following variable should be private but there are public for
  // simplicity
  MPI_Comm _comm;
  dealii::parallel::distributed::Triangulation<dim> _triangulation;
  dealii::FE_Q<dim> _fe;
  dealii::DoFHandler<dim> _dof_handler;
  dealii::IndexSet _locally_owned_dofs;
  dealii::IndexSet _locally_relevant_dofs;
  dealii::AffineConstraints<double> _constraints;
  LaplaceOperator<dim, fe_degree, ScalarType> _laplace_operator;
  dealii::LinearAlgebra::distributed::Vector<ScalarType> _solution;
  dealii::LinearAlgebra::distributed::Vector<ScalarType> _system_rhs;
};

template <int dim, int fe_degree, typename ScalarType>
LaplaceMatrixFree<dim, fe_degree, ScalarType>::LaplaceMatrixFree(
    MPI_Comm const &comm)
    : _comm(comm), _triangulation(_comm), _fe(fe_degree),
      _dof_handler(_triangulation)
{
}

template <int dim, int fe_degree, typename ScalarType>
template <typename MaterialPropertyType>
void LaplaceMatrixFree<dim, fe_degree, ScalarType>::setup_system(
    boost::property_tree::ptree const &ptree,
    MaterialPropertyType const &material_property)
{
  std::string const mesh = ptree.get("mesh", "hyper_cube");
  if (mesh == "hyper_ball")
    dealii::GridGenerator::hyper_ball(_triangulation);
  else
    dealii::GridGenerator::hyper_cube(_triangulation);

  _triangulation.refine_global(ptree.get("n_refinements", 3));

  if (ptree.get("distort_random", false))
    dealii::GridTools::distort_random(0.2, _triangulation);

  _dof_handler.distribute_dofs(_fe);

  std::string const reordering = ptree.get("reordering", "None");
  if (reordering == "Reverse Cuthill-McKee")
    dealii::DoFRenumbering::Cuthill_McKee(_dof_handler, true);
  else if (reordering == "King")
    dealii::DoFRenumbering::boost::king_ordering(_dof_handler);
  else if (reordering == "Reverse minimum degree")
    dealii::DoFRenumbering::boost::minimum_degree(_dof_handler, true);
  else if (reordering == "Hierarchical")
    dealii::DoFRenumbering::hierarchical(_dof_handler);

  // Get the IndexSets
  _locally_owned_dofs = _dof_handler.locally_owned_dofs();
  dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler,
                                                  _locally_relevant_dofs);

  // Compute the constraints
  _constraints.clear();
  _constraints.reinit(_locally_relevant_dofs);
  dealii::DoFTools::make_hanging_node_constraints(_dof_handler, _constraints);
  dealii::VectorTools::interpolate_boundary_values(
      _dof_handler, 0, dealii::Functions::ZeroFunction<dim>(), _constraints);
  _constraints.close();

  // Initialize the MatrixFree object
  typename dealii::MatrixFree<dim, ScalarType>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
      dealii::MatrixFree<dim, ScalarType>::AdditionalData::none;
  additional_data.mapping_update_flags = dealii::update_gradients |
                                         dealii::update_JxW_values |
                                         dealii::update_quadrature_points;
  std::shared_ptr<dealii::MatrixFree<dim, ScalarType>> mf_storage(
      new dealii::MatrixFree<dim, ScalarType>());
  mf_storage->reinit(_dof_handler, _constraints,
                     dealii::QGauss<1>(fe_degree + 1), additional_data);
  _laplace_operator.initialize(mf_storage);

  // Resize the vectors
  _laplace_operator.initialize_dof_vector(_solution);
  _laplace_operator.initialize_dof_vector(_system_rhs);

  _laplace_operator.evaluate_coefficient(material_property);
}

template <int dim, int fe_degree, typename ScalarType>
template <typename SourceType>
void LaplaceMatrixFree<dim, fe_degree, ScalarType>::assemble_rhs(
    SourceType const &source)
{
  _system_rhs = 0.;
  dealii::FEEvaluation<dim, fe_degree> fe_eval(
      *_laplace_operator.get_matrix_free());
  unsigned int const n_macro_cells =
      _laplace_operator.get_matrix_free()->n_macro_cells();
  bool constexpr integrate_values = true;
  bool constexpr integrate_gradients = false;
  for (unsigned int cell = 0; cell < n_macro_cells; ++cell)
  {
    fe_eval.reinit(cell);
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      fe_eval.submit_value(source.value(fe_eval.quadrature_point(q)), q);
    fe_eval.integrate(integrate_values, integrate_gradients);
    fe_eval.distribute_local_to_global(_system_rhs);
  }
  _system_rhs.compress(dealii::VectorOperation::add);
}

template <int dim, int fe_degree, typename ScalarType>
template <typename PreconditionerType>
void LaplaceMatrixFree<dim, fe_degree, ScalarType>::solve(
    PreconditionerType &preconditioner)
{
  dealii::SolverControl solver_control(_dof_handler.n_dofs(),
                                       1e-12 * _system_rhs.l2_norm());
  dealii::SolverCG<dealii::LinearAlgebra::distributed::Vector<double>> cg(
      solver_control);
  _constraints.set_zero(_solution);

  cg.solve(_laplace_operator, _solution, _system_rhs, preconditioner);

  if (dealii::Utilities::MPI::this_mpi_process(_comm) == 0)
    std::cout << "Solved in " << solver_control.last_step() << " iterations."
              << std::endl;

  _constraints.distribute(_solution);
}

template <int dim, int fe_degree, typename ScalarType>
double LaplaceMatrixFree<dim, fe_degree, ScalarType>::compute_error(
    dealii::Function<dim> const &exact_solution)
{
  dealii::QGauss<dim> const quadrature(fe_degree + 1);

  // We need a regular ghosted vector but for some reason we need to first get a
  // non ghosted vector and then, we can get a ghosted vector.
  dealii::LinearAlgebra::distributed::Vector<ScalarType>
      locally_relevant_solution(_locally_owned_dofs, _locally_relevant_dofs,
                                _comm);
  dealii::LinearAlgebra::distributed::Vector<ScalarType> sol(
      _locally_owned_dofs, _comm);
  sol = _solution;
  locally_relevant_solution = sol;

  dealii::Vector<double> difference(
      _dof_handler.get_triangulation().n_active_cells());
  dealii::VectorTools::integrate_difference(
      _dof_handler, locally_relevant_solution, exact_solution, difference,
      quadrature, dealii::VectorTools::L2_norm);

  return dealii::VectorTools::compute_global_error(
      _dof_handler.get_triangulation(), difference,
      dealii::VectorTools::L2_norm);
}

#endif
