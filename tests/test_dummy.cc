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

#define BOOST_TEST_MODULE dummy

#include "main.cc"

#include "mfmg/dummy.hpp"

BOOST_AUTO_TEST_CASE(hyper_cube)
{
  Dummy dummy;
  dealii::Triangulation<2> tria;
  dummy.generate_mesh(tria);

  BOOST_TEST(tria.n_active_cells() == 4);
}
