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

#include <deal.II/dofs/dof_handler.h>

#include <array>

namespace mfmg
{
template <int dim, typename VectorType>
class AMGe
{
public:
  AMGe(dealii::DoFHandler<dim> &dof_handler);

  void build_agglomerate(std::array<unsigned int, dim> const &agglomerate_dim);

private:
  dealii::DoFHandler<dim> &_dof_handler;
};
}
