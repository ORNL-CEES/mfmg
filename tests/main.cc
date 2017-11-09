/*************************************************************************
 * Copyright (c) 2017 by the mfmg authors                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *************************************************************************/

#define BOOST_TEST_NO_MAIN
#include <deal.II/base/mpi.h>
#include <boost/test/unit_test.hpp>

bool init_function() 
{
  return true;
}

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 
       dealii::numbers::invalid_unsigned_int);

  return boost::unit_test::unit_test_main(&init_function, argc, argv);
}
