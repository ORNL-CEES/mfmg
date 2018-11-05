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

#ifndef MFMG_DEALII_MESH_EVALUATOR_HPP
#define MFMG_DEALII_MESH_EVALUATOR_HPP

#include <mfmg/common/exceptions.hpp>
#include <mfmg/common/mesh_evaluator.hpp>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

namespace mfmg
{
template <int dim>
class DealIIMeshEvaluator : public MeshEvaluator
{
public:
  DealIIMeshEvaluator(
      dealii::DoFHandler<dim> &dof_handler,
      dealii::AffineConstraints<double> &constraints,
      std::string const &mesh_evaluator_type = "DealIIMeshEvaluator");

  int get_dim() const override final;

  std::string get_mesh_evaluator_type() const override final;

  // For deal.II, because of the way it deals with hanging nodes and
  // Dirichlet b.c., we need to zero out the initial guess values
  // corresponding to those. Otherwise, it may cause issues with spurious
  // modes and some scaling difficulties. If we use `apply` for deal.II, we
  // don't need this as we can apply the constraints immediately after
  // applying the matrix and before any norms and dot products.
  // In addition, this has to work only for ARPACK with dealii::Vector<double>
  void set_initial_guess(dealii::AffineConstraints<double> &constraints,
                         dealii::Vector<double> &x) const;

  virtual void evaluate_agglomerate(dealii::DoFHandler<dim> &,
                                    dealii::AffineConstraints<double> &,
                                    dealii::SparsityPattern &,
                                    dealii::SparseMatrix<double> &) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  virtual void evaluate_global(dealii::DoFHandler<dim> &,
                               dealii::AffineConstraints<double> &,
                               dealii::TrilinosWrappers::SparseMatrix &) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  dealii::DoFHandler<dim> &get_dof_handler();

  dealii::AffineConstraints<double> &get_constraints();

protected:
  dealii::DoFHandler<dim> &_dof_handler;
  dealii::AffineConstraints<double> &_constraints;
  std::string const _mesh_evaluator_type;
};
} // namespace mfmg

#endif
