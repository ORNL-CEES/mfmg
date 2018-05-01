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

#ifndef MFMG_DEALII_MESH_HPP
#define MFMG_DEALII_MESH_HPP

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>

namespace mfmg
{
template <int dim>
struct DealIIMesh
{
  DealIIMesh(dealii::DoFHandler<dim> &dof_handler,
             dealii::ConstraintMatrix &constraints)
      : _dof_handler(dof_handler), _constraints(constraints)
  {
  }

  static constexpr int dimension() { return dim; }
  dealii::DoFHandler<dim> &_dof_handler;
  dealii::ConstraintMatrix &_constraints;
};
}

#endif
