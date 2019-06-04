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

#ifndef MFMG_TEST_HIERARCHY_HELPERS_DEVICE_CUH
#define MFMG_TEST_HIERARCHY_HELPERS_DEVICE_CUH

#include <deal.II/base/function.h>

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
class ConstantMaterialProperty final : public dealii::Function<dim>
{
public:
  ConstantMaterialProperty() = default;

  virtual ~ConstantMaterialProperty() override = default;

  virtual double value(dealii::Point<dim> const &,
                       unsigned int const = 0) const override
  {
    return 1.;
  }
};

template <int dim>
class LinearXMaterialProperty final : public dealii::Function<dim>
{
public:
  LinearXMaterialProperty() = default;

  virtual ~LinearXMaterialProperty() override = default;

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override
  {
    return 1. + p[0];
  }
};

template <int dim>
class LinearMaterialProperty final : public dealii::Function<dim>
{
public:
  LinearMaterialProperty() = default;

  virtual ~LinearMaterialProperty() override = default;

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override
  {
    double value = 1.;
    for (unsigned int d = 0; d < dim; ++d)
      value += (1. + d) * p[d];

    return value;
  }
};

template <int dim>
class DiscontinuousMaterialProperty : public dealii::Function<dim>
{
public:
  DiscontinuousMaterialProperty() = default;

  virtual ~DiscontinuousMaterialProperty() override = default;

  virtual double value(dealii::Point<dim> const &p,
                       unsigned int const = 0) const override final

  {
    double value = 10.;
    for (unsigned int d = 0; d < dim; ++d)
      if (p[d] > 0.5)
        value *= value;

    return value;
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

template <int dim>
class TestMeshEvaluator final : public mfmg::CudaMeshEvaluator<dim>
{
public:
  TestMeshEvaluator(MPI_Comm comm, dealii::DoFHandler<dim> &dof_handler,
                    dealii::AffineConstraints<double> &constraints,
                    unsigned int fe_degree,
                    dealii::TrilinosWrappers::SparseMatrix const &matrix,
                    std::shared_ptr<dealii::Function<dim>> material_property,
                    mfmg::CudaHandle &cuda_handle)
      : mfmg::CudaMeshEvaluator<dim>(cuda_handle, dof_handler, constraints),
        _comm(comm), _fe_degree(fe_degree), _matrix(matrix),
        _material_property(material_property)
  {
  }

  virtual ~TestMeshEvaluator() override = default;

  virtual dealii::LinearAlgebra::distributed::Vector<double>
  get_locally_relevant_diag() const override
  {
    dealii::IndexSet locally_owned_dofs =
        _matrix.locally_owned_domain_indices();
    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(this->_dof_handler,
                                                    locally_relevant_dofs);
    dealii::LinearAlgebra::distributed::Vector<double>
        locally_owned_global_diag(locally_owned_dofs, _comm);
    for (auto const val : locally_owned_dofs)
      locally_owned_global_diag[val] = _matrix.diag_element(val);
    locally_owned_global_diag.compress(dealii::VectorOperation::insert);

    dealii::LinearAlgebra::distributed::Vector<double>
        locally_relevant_global_diag(locally_owned_dofs, locally_relevant_dofs,
                                     _comm);
    locally_relevant_global_diag = locally_owned_global_diag;

    return locally_relevant_global_diag;
  }

  void evaluate_global(
      dealii::DoFHandler<dim> &, dealii::AffineConstraints<double> &,
      mfmg::SparseMatrixDevice<double> &system_matrix) const override
  {
    system_matrix = std::move(mfmg::convert_matrix(_matrix));
    system_matrix.cusparse_handle = this->_cuda_handle.cusparse_handle;
    cusparseStatus_t cusparse_error_code;
    cusparse_error_code = cusparseCreateMatDescr(&system_matrix.descr);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatType(system_matrix.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatIndexBase(system_matrix.descr, CUSPARSE_INDEX_BASE_ZERO);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  }

  void evaluate_agglomerate(
      dealii::DoFHandler<dim> &dof_handler,
      dealii::AffineConstraints<double> &constraints,
      mfmg::SparseMatrixDevice<double> &system_matrix) const override
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
    dealii::SparsityPattern agg_system_sparsity_pattern;
    agg_system_sparsity_pattern.copy_from(dsp);
    dealii::SparseMatrix<double> agg_system_matrix(agg_system_sparsity_pattern);

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
                                             agg_system_matrix);
    }

    system_matrix = std::move(mfmg::convert_matrix(agg_system_matrix));
    system_matrix.cusparse_handle = this->_cuda_handle.cusparse_handle;
    cusparseStatus_t cusparse_error_code;
    cusparse_error_code = cusparseCreateMatDescr(&system_matrix.descr);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatType(system_matrix.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code =
        cusparseSetMatIndexBase(system_matrix.descr, CUSPARSE_INDEX_BASE_ZERO);
    mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  }

private:
  MPI_Comm _comm;
  unsigned const _fe_degree;
  dealii::TrilinosWrappers::SparseMatrix const &_matrix;
  std::shared_ptr<dealii::Function<dim>> _material_property;
};

#endif
