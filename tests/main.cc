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

#define BOOST_TEST_NO_MAIN

#include <deal.II/base/mpi.h>

#include <boost/test/unit_test.hpp>

bool init_function() { return true; }

int main(int argc, char *argv[])
{
  // Set the maximum number of threads used to the minimum of the number of
  // cores reported by TBB and the environment variable DEAL_II_NUM_THREADS.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(
      argc, argv, dealii::numbers::invalid_unsigned_int);

  return boost::unit_test::unit_test_main(&init_function, argc, argv);
}
