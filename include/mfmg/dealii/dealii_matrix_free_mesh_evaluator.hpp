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

/**
 * An interface class the user should derive from when defining an operator to
 * be used in a matrix-free context.
 */
template <int dim>
class DealIIMatrixFreeMeshEvaluator : public DealIIMeshEvaluator<dim>
{
public:
  using size_type = dealii::types::global_dof_index;

  /**
   * The dimension of the underlying mesh.
   */
  static int constexpr _dim = dim;

  /**
   * The constructor is supposed to initialize the data required for performing
   * operator evaluations on the global finite element space given by @p
   * dof_handler and @p constraints.
   */
  DealIIMatrixFreeMeshEvaluator(dealii::DoFHandler<dim> &dof_handler,
                                dealii::AffineConstraints<double> &constraints);

  /**
   * Destructor.
   */
  virtual ~DealIIMatrixFreeMeshEvaluator() override = default;

  /**
   * Create a deep copy of this class such that initializing on another
   * agglomerate works.
   */
  virtual std::unique_ptr<DealIIMatrixFreeMeshEvaluator> clone() const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
    return std::make_unique<DealIIMatrixFreeMeshEvaluator>(*this);
  }

  /**
   * Return the class name as std::string.
   */
  std::string get_mesh_evaluator_type() const override final;

  std::shared_ptr<dealii::LinearAlgebra::distributed::Vector<double>>
  build_range_vector() const;

  /**
   * Return the dimension of the finite element space.
   */
  dealii::types::global_dof_index m() const;

  /**
   * Initialize the operator for a given agglomerate described by @p
   * dof_handler.
   * @p dof_handler is expected to be initialized with a dealii::FiniteElement
   * object in this call.
   */
  virtual void matrix_free_initialize_agglomerate(
      dealii::DoFHandler<dim> & /*dof_handler*/) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  /**
   * Evaluate the operator on the agglomerate this object was initialized on in
   * matrix_free_initialize_agglomerate().
   */
  virtual void
  matrix_free_evaluate_agglomerate(dealii::Vector<double> const & /*src*/,
                                   dealii::Vector<double> & /*dst*/) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  /**
   * Return the diagonal of the matrix the agglomerate operator conrresponds to.
   */
  virtual std::vector<double> matrix_free_get_agglomerate_diagonal(
      dealii::AffineConstraints<double> & /*constraints*/) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return std::vector<double>();
  }

  /**
   *  Evaluate the operator on the global mesh.
   */
  virtual void matrix_free_evaluate_global(
      dealii::LinearAlgebra::distributed::Vector<double> const & /*src*/,
      dealii::LinearAlgebra::distributed::Vector<double> & /*dst*/) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  /**
   * Return the inverse of the diagonal of the matrix the global operator
   * corresponds to. Constrained degrees of freedom are set to zero.
   */
  virtual std::shared_ptr<dealii::DiagonalMatrix<
      dealii::LinearAlgebra::distributed::Vector<double>>>
  matrix_free_get_diagonal_inverse() const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return nullptr;
  }

  /**
   * Return the diagonal of the matrix the global operator corresponds to.
   * Constrained degrees of freedom are set to zero.
   */
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
