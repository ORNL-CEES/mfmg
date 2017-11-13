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

#define BOOST_TEST_MODULE laplace

#include "laplace.hpp"
#include "main.cc"
#include <cstdio>
#include <deal.II/lac/trilinos_precondition.h>

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE(laplace_2d)
{
  boost::mpi::communicator world;

  Laplace<2, dealii::TrilinosWrappers::MPI::Vector> laplace(world, 2);
  laplace.setup_system();
  laplace.assemble_system();
  dealii::TrilinosWrappers::PreconditionSSOR preconditioner;
  dealii::TrilinosWrappers::MPI::Vector solution =
      laplace.solve(preconditioner);

  double ref = 0.660145;
  BOOST_TEST(solution.l2_norm() == ref, tt::tolerance(1.e-6));

  laplace.output_results();
  // Remove output file
  if (world.rank() == 0)
  {
    BOOST_TEST(std::remove("solution.pvtu") == 0);
    for (int i = 0; i < world.size(); ++i)
      BOOST_TEST(
          std::remove(("solution-" + std::to_string(i) + ".vtu").c_str()) == 0);
  }
}
