/**************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                            *
 * All rights reserved.                                                   *
 *                                                                        *
 * This file is part of the mfmg library. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the  *
 * top-level directory                                                    *
 *                                                                        *
 * SPDX-License-Identifier: BSD-3-Clause                                  *
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
#include "laplace_matrix_free.hpp"

namespace bdata = boost::unit_test::data;
namespace tt = boost::test_tools;

template <int dim>
class Source final : public dealii::Function<dim>
{
public:
  Source() = default;

  virtual ~Source() override = default;

  virtual double value(dealii::Point<dim> const &,
                       unsigned int const = 0) const override
  {
    return 0.;
  }
};

template <int dim>
class Coefficient : public dealii::Function<dim>
{
public:
  virtual ~Coefficient() override = default;

  virtual dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &p,
        unsigned int const = 0) const = 0;

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const component = 0) const override = 0;
};

template <int dim>
class ConstantMaterialProperty final : public Coefficient<dim>
{
public:
  ConstantMaterialProperty() = default;

  virtual ~ConstantMaterialProperty() override = default;

  virtual dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &,
        unsigned int const = 0) const override
  {
    return dealii::make_vectorized_array<double>(1.);
  }

  virtual double value(dealii::Point<dim> const &,
                       unsigned int const = 0) const override
  {
    return 1.;
  }
};

template <int dim>
class LinearXMaterialProperty final : public Coefficient<dim>
{
public:
  LinearXMaterialProperty() = default;

  virtual ~LinearXMaterialProperty() override = default;

  virtual dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &p,
        unsigned int const = 0) const override
  {
    auto const one = dealii::make_vectorized_array<double>(1.);
    return one + std::abs(p[0]);
  }

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override
  {
    return 1. + std::abs(p[0]);
  }
};

template <int dim>
class LinearMaterialProperty final : public Coefficient<dim>
{
public:
  LinearMaterialProperty() = default;

  virtual ~LinearMaterialProperty() override = default;

  virtual dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &p,
        unsigned int const = 0) const override
  {
    auto const one = dealii::make_vectorized_array<double>(1.);
    auto val = one;
    for (unsigned int d = 0; d < dim; ++d)
      val += (one + static_cast<double>(d) * one) * std::abs(p[d]);

    return val;
  }

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override
  {
    double val = 1.;
    for (unsigned int d = 0; d < dim; ++d)
      val += (1. + d) * std::abs(p[d]);

    return val;
  }
};

template <int dim>
class DiscontinuousMaterialProperty final : public Coefficient<dim>
{
public:
  DiscontinuousMaterialProperty() = default;

  virtual ~DiscontinuousMaterialProperty() override = default;

  virtual dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &p,
        unsigned int const = 0) const override
  {
    auto const one = dealii::make_vectorized_array<double>(1.);
    unsigned int dim_scale = 0;
    for (unsigned int d = 0; d < dim; ++d)
      dim_scale += static_cast<unsigned int>(std::floor(p[d][0] * 100)) % 2;

    return (dim_scale == dim ? 100. * one : 10. * one);
  }

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override

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
  static std::shared_ptr<Coefficient<dim>>
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
class TestMeshEvaluator final : public MeshEvaluator
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

  virtual ~TestMeshEvaluator() override = default;

  virtual void evaluate_global(
      dealii::DoFHandler<dim> &, dealii::AffineConstraints<double> &,
      dealii::TrilinosWrappers::SparseMatrix &system_matrix) const override
  {
    // TODO this is pretty expansive, we should use a shared pointer
    system_matrix.copy_from(_matrix);
  }

  virtual void evaluate_agglomerate(
      dealii::DoFHandler<dim> &dof_handler,
      dealii::AffineConstraints<double> &constraints,
      dealii::SparsityPattern &system_sparsity_pattern,
      dealii::SparseMatrix<double> &system_matrix) const override
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

template <int dim, int fe_degree, typename ScalarType>
class TestMFMeshEvaluator final
    : public mfmg::DealIIMatrixFreeMeshEvaluator<dim>
{
public:
  TestMFMeshEvaluator(
      dealii::DoFHandler<dim> &dof_handler,
      dealii::AffineConstraints<double> &constraints,
      LaplaceOperator<dim, fe_degree, ScalarType> &laplace_operator,
      std::shared_ptr<Coefficient<dim>> material_property)
      : mfmg::DealIIMatrixFreeMeshEvaluator<dim>(dof_handler, constraints),
        _material_property(material_property), _fe(fe_degree),
        _laplace_operator(laplace_operator)
  {
  }

  // We need a copy constructor since the evaluator has to cloned for each
  // agglomerate. Here, we just copy the minimum number of member variables
  // required. In particular, we should not need to copy mutable members since
  // they are set up in matrix_free_initialize_agglomerate only.
  TestMFMeshEvaluator(
      TestMFMeshEvaluator<dim, fe_degree, ScalarType> const &_other_evaluator)
      : mfmg::DealIIMatrixFreeMeshEvaluator<dim>(*this),
        _material_property(_other_evaluator._material_property),
        _fe(_other_evaluator._fe),
        _laplace_operator(_other_evaluator._laplace_operator)
  {
  }

  virtual ~TestMFMeshEvaluator() override = default;

  virtual std::unique_ptr<mfmg::DealIIMatrixFreeMeshEvaluator<dim>>
  clone() const override
  {
    return std::make_unique<TestMFMeshEvaluator>(*this);
  }

  virtual void
  matrix_free_evaluate_agglomerate(dealii::Vector<double> const &src,
                                   dealii::Vector<double> &dst) const override
  {
    // Unfortunately, dealii::MatrixFreeOperators::Base only supports
    // dealii::LinearAlgebra::distributed::Vector so unless we duplicate a lot
    // of code, we need copy src and dst.
    std::copy(src.begin(), src.end(), distributed_src.begin());
    _agg_laplace_operator->vmult(distributed_dst, distributed_src);
    std::copy(distributed_dst.begin(), distributed_dst.end(), dst.begin());
  }

  virtual std::vector<double> matrix_free_get_agglomerate_diagonal(
      dealii::AffineConstraints<double> &constraints) const override
  {
    constraints.copy_from(_agg_constraints);

    auto diag_matrix = _agg_laplace_operator->get_matrix_diagonal();
    auto diag_dealii_vector = diag_matrix->get_vector();

    return std::vector<double>(diag_dealii_vector.begin(),
                               diag_dealii_vector.end());
  }

  virtual void matrix_free_evaluate_global(
      dealii::LinearAlgebra::distributed::Vector<double> const &src,
      dealii::LinearAlgebra::distributed::Vector<double> &dst) const override
  {
    _laplace_operator.vmult(dst, src);
  }

  virtual std::shared_ptr<dealii::DiagonalMatrix<
      dealii::LinearAlgebra::distributed::Vector<double>>>
  matrix_free_get_diagonal_inverse() const override
  {
    return _laplace_operator.get_matrix_diagonal_inverse();
  }

  virtual dealii::LinearAlgebra::distributed::Vector<double>
  get_diagonal() override
  {
    auto vector = _laplace_operator.get_matrix_diagonal()->get_vector();
    vector.update_ghost_values();

    return vector;
  }

  virtual void matrix_free_initialize_agglomerate(
      dealii::DoFHandler<dim> &dof_handler) const override
  {
    // FIXME dof_handler should be const and initialized somewhere else
    dof_handler.distribute_dofs(_fe);

    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                    locally_relevant_dofs);
    // Compute the constraints
    _agg_constraints.clear();
    _agg_constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                    _agg_constraints);
    dealii::VectorTools::interpolate_boundary_values(
        dof_handler, 1, dealii::Functions::ZeroFunction<dim>(),
        _agg_constraints);
    _agg_constraints.close();

    // Initialize the MatrixFree object
    typename dealii::MatrixFree<dim, ScalarType>::AdditionalData
        additional_data;
    additional_data.tasks_parallel_scheme =
        dealii::MatrixFree<dim, ScalarType>::AdditionalData::none;
    additional_data.mapping_update_flags = dealii::update_gradients |
                                           dealii::update_JxW_values |
                                           dealii::update_quadrature_points;
    std::shared_ptr<dealii::MatrixFree<dim, ScalarType>> mf_storage(
        new dealii::MatrixFree<dim, ScalarType>());
    mf_storage->reinit(
        dof_handler, _agg_constraints, dealii::QGauss<1>(fe_degree + 1),
        additional_data,
        &(_laplace_operator.get_matrix_free()->get_raw_shape_info()));

    _agg_laplace_operator =
        std::make_unique<LaplaceOperator<dim, fe_degree, ScalarType>>();
    _agg_laplace_operator->initialize(mf_storage);
    _agg_laplace_operator->evaluate_coefficient(*_material_property);
    _agg_laplace_operator->compute_diagonal();

    distributed_dst.reinit(dof_handler.n_dofs());
    distributed_src.reinit(dof_handler.n_dofs());
  }

private:
  std::shared_ptr<Coefficient<dim>> _material_property;
  dealii::FE_Q<dim> _fe;
  mutable dealii::AffineConstraints<double> _agg_constraints;
  LaplaceOperator<dim, fe_degree, ScalarType> &_laplace_operator;
  mutable std::unique_ptr<LaplaceOperator<dim, fe_degree, ScalarType>>
      _agg_laplace_operator;
  mutable dealii::LinearAlgebra::distributed::Vector<ScalarType>
      distributed_dst;
  mutable dealii::LinearAlgebra::distributed::Vector<ScalarType>
      distributed_src;
};

#endif // #ifdef MFMG_TEST_HIERARCHY_HELPERS_HPP
