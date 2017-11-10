/*************************************************************************
 * Copyright (c) 2017 by the mfmg authors                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *************************************************************************/

#include <deal.II/grid/tria.h>

class Dummy
{
public:
  Dummy() = default;

  void generate_mesh(dealii::Triangulation<2> &tria);
};
