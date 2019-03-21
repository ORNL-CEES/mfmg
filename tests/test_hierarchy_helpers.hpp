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

#ifndef MFMG_TEST_HIERARCHY_HELPERS_HPP
#define MFMG_TEST_HIERARCHY_HELPERS_HPP

#include <mfmg/common/hierarchy.hpp>
#include <mfmg/dealii/dealii_matrix_free_mesh_evaluator.hpp>
#include <mfmg/dealii/dealii_mesh_evaluator.hpp>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_linear_operator.h>
#include <deal.II/lac/trilinos_vector.h>

#include <boost/property_tree/info_parser.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>

#include <random>

#include "laplace.hpp"

namespace bdata = boost::unit_test::data;
namespace tt = boost::test_tools;

template <int dim>
class Source : public dealii::Function<dim>
{
public:
  Source() = default;

  virtual double value(dealii::Point<dim> const &,
                       unsigned int const = 0) const override final
  {
    return 0.;
  }
};

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
class LinearXMaterialProperty : public dealii::Function<dim>
{
public:
  LinearXMaterialProperty() = default;

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override final
  {
    return 1. + std::abs(p[0]);
  }
};

template <int dim>
class LinearMaterialProperty : public dealii::Function<dim>
{
public:
  LinearMaterialProperty() = default;

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override final
  {
    double value = 1.;
    for (unsigned int d = 0; d < dim; ++d)
      value += (1. + d) * std::abs(p[d]);

    return value;
  }
};

template <int dim>
class DiscontinuousMaterialProperty : public dealii::Function<dim>
{
public:
  DiscontinuousMaterialProperty() = default;

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override final

  {
    unsigned int dim_scale = 0;
    for (unsigned int d = 0; d < dim; ++d)
      dim_scale += static_cast<unsigned int>(std::floor(p[d] * 100)) % 2;

    return (dim_scale == dim ? 100. : 10.);
  }
};

template <int dim>
class MaterialPropertyFactory
{
public:
  static std::shared_ptr<dealii::Function<dim>>
  create_material_property(std::string const &material_type)
  {
    if (material_type == "constant")
      return std::make_shared<ConstantMaterialProperty<dim>>();
    else if (material_type == "linear_x")
      return std::make_shared<LinearXMaterialProperty<dim>>();
    else if (material_type == "linear")
      return std::make_shared<LinearMaterialProperty<dim>>();
    else if (material_type == "discontinuous")
      return std::make_shared<DiscontinuousMaterialProperty<dim>>();
    else
    {
      mfmg::ASSERT_THROW_NOT_IMPLEMENTED();

      return nullptr;
    }
  }
};

template <typename MeshEvaluator>
class TestMeshEvaluator : public MeshEvaluator
{
public:
  static int constexpr dim = MeshEvaluator::_dim;
  TestMeshEvaluator(dealii::DoFHandler<dim> &dof_handler,
                    dealii::AffineConstraints<double> &constraints,
                    unsigned int fe_degree,
                    dealii::TrilinosWrappers::SparseMatrix const &matrix,
                    std::shared_ptr<dealii::Function<dim>> material_property)
      : MeshEvaluator(dof_handler, constraints), _fe_degree(fe_degree),
        _matrix(matrix), _material_property(material_property)
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
    unsigned int const fe_degree = _fe_degree;
    dealii::FE_Q<dim> fe(fe_degree);
    dof_handler.distribute_dofs(fe);

    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                    locally_relevant_dofs);

    // Compute the constraints
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    dealii::VectorTools::interpolate_boundary_values(
        dof_handler, 1, dealii::Functions::ZeroFunction<dim>(), constraints);
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
      {
        double const diffusion_coefficient =
            _material_property->value(fe_values.quadrature_point(q_point));
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) +=
                diffusion_coefficient * fe_values.shape_grad(i, q_point) *
                fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, local_dof_indices,
                                             system_matrix);
    }
  }

private:
  unsigned const _fe_degree;
  dealii::TrilinosWrappers::SparseMatrix const &_matrix;
  std::shared_ptr<dealii::Function<dim>> _material_property;
};

#endif // #ifdef MFMG_TEST_HIERARCHY_HELPERS_HPP
