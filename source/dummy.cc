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

#include "mfmg/dummy.hpp"
#include <deal.II/grid/grid_generator.h>

void Dummy::generate_mesh(dealii::Triangulation<2> &tria)
{
  dealii::GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
}
