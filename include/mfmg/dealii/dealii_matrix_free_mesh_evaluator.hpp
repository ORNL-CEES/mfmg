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

#ifndef MFMG_DEALII_MATRIX_FREE_MESH_EVALUATOR_HPP
#define MFMG_DEALII_MATRIX_FREE_MESH_EVALUATOR_HPP

#include <mfmg/dealii/dealii_mesh_evaluator.hpp>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <type_traits>

namespace mfmg
{
template <int dim>
class DealIIMatrixFreeMeshEvaluator : public DealIIMeshEvaluator<dim>
{
public:
  using size_type = dealii::types::global_dof_index;
  static int constexpr _dim = dim;

  DealIIMatrixFreeMeshEvaluator(dealii::DoFHandler<dim> &dof_handler,
                                dealii::AffineConstraints<double> &constraints);

  std::string get_mesh_evaluator_type() const override final;

  std::shared_ptr<dealii::LinearAlgebra::distributed::Vector<double>>
  build_range_vector() const;

  dealii::types::global_dof_index m() const;

  virtual void
  matrix_free_evaluate_agglomerate(dealii::DoFHandler<dim> & /*dof_handler*/,
                                   dealii::Vector<double> const & /*src*/,
                                   dealii::Vector<double> & /*dst*/) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  virtual std::vector<double> matrix_free_get_agglomerate_diagonal(
      dealii::DoFHandler<dim> & /*dof_handler*/,
      dealii::AffineConstraints<double> & /*constraints*/) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return std::vector<double>();
  }

  virtual void matrix_free_evaluate_global(
      dealii::LinearAlgebra::distributed::Vector<double> const & /*src*/,
      dealii::LinearAlgebra::distributed::Vector<double> & /*dst*/) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  virtual std::shared_ptr<dealii::DiagonalMatrix<
      dealii::LinearAlgebra::distributed::Vector<double>>>
  matrix_free_get_diagonal_inverse() const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return nullptr;
  }

  virtual dealii::LinearAlgebra::distributed::Vector<double>
  get_diagonal() override
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return dealii::LinearAlgebra::distributed::Vector<double>();
  }
};

template <int dim>
struct is_matrix_free<DealIIMatrixFreeMeshEvaluator<dim>> : std::true_type
{
};

} // namespace mfmg

#endif
