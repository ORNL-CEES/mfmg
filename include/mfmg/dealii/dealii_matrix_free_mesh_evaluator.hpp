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

#ifndef MFMG_DEALII_MATRIX_FREE_MESH_EVALUATOR_HPP
#define MFMG_DEALII_MATRIX_FREE_MESH_EVALUATOR_HPP

#include <mfmg/dealii/dealii_mesh_evaluator.hpp>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

namespace mfmg
{
template <int dim>
class DealIIMatrixFreeMeshEvaluator : public DealIIMeshEvaluator<dim>
{
public:
  DealIIMatrixFreeMeshEvaluator(dealii::DoFHandler<dim> &dof_handler,
                                dealii::AffineConstraints<double> &constraints);

  std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> get_matrix();

  void
  vmult(dealii::LinearAlgebra::distributed::Vector<double> &dst,
        dealii::LinearAlgebra::distributed::Vector<double> const &src) const;

  std::shared_ptr<dealii::LinearAlgebra::distributed::Vector<double>>
  build_range_vector() const;

private:
  std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> _sparse_matrix;
};
} // namespace mfmg

#endif
