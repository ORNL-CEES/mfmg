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

#define BOOST_TEST_MODULE lanczos

#include "mfmg/dealii/lanczos.templates.hpp"

#include <cmath>
#include <cstdio>

#include "lanczos_simpleop.templates.hpp"
#include "lanczos_simplevector.templates.hpp"
#include "main.cc"

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(lanczos)
{
  using namespace mfmg::lanczos;

  typedef double ScalarType;
  typedef SimpleVector<ScalarType> VectorType;
  typedef SimpleOp<VectorType> OperatorType;
  typedef Lanczos<OperatorType> Solver_t;

  const size_t n = 1000;
  const int multiplicity = 2;

  OperatorType op(n, multiplicity);

  // std::cout << "Test matrix is a diagonal matrix with eigenvalues 1, 2, ...
  // ."
  //    << std::endl;
  // std::cout << "Matrix dimension: " << n << " multiplicity of all
  // eigenvalues: "
  //    << multiplicity << "." << std::endl;

  const int num_evecs_per_cycle = 2;
  const int num_cycles = 3;

  // std::cout << "Number of Lanczos solves to be performed in sequence: "
  //    << num_cycles << std::endl;
  // std::cout << "Number of eigenpairs computed per Lanczos solve: "
  //    << num_evecs_per_cycle << std::endl;
  // std::cout << "Each Lanczos solve will deflate "
  //             "previously computed eigenvectors." << std::endl;

  const int maxit = 200;
  const double tol = 1.e-2;
  const double percent_overshoot = 5;
  const int verbosity = 0; // 1;

  // std::cout << "Maximum iterations for each Lanczos solve: " << maxit
  //    << std::endl;
  // std::cout << "Convergence tolerance: " << tol << std::endl;
  // std::cout << "Percent overshoot num iterations allowed by stopping test: "
  //    << percent_overshoot << std::endl;
  // std::cout << std::endl;

  boost::property_tree::ptree lanczos_params;
  lanczos_params.put("is_deflated", true);
  lanczos_params.put("num_eigenpairs_per_cycle", num_evecs_per_cycle);
  lanczos_params.put("num_cycles", num_cycles);
  lanczos_params.put("max_iterations", maxit);
  lanczos_params.put("tolerance", tol);
  lanczos_params.put("percent_overshoot", percent_overshoot);
  lanczos_params.put("verbosity", verbosity);
  Solver_t solver(op, lanczos_params);

  solver.solve();

  std::cout << "Final approximate eigenvalues: " << std::endl;

  for (int i = 0; i < solver.num_evecs(); ++i)
  {
    std::cout.width(4);
    std::cout.precision(8);
    std::cout << std::fixed << solver.get_eval(i) << " ";
  }
  std::cout << std::endl;

  //-----

  unsigned int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  BOOST_TEST(rank == 0); // only need to test on 1 rank

  // Loop to ensure each Ritz value is near an eigenvalue.
  // TODO: make a more complete check of correctness of result.

  const int num_ritzvals = solver.num_evecs();
  int num_passed = 0;
  const double tol_test = 1. * tol; // this may need adjustment

  for (int i = 0; i < num_ritzvals; ++i)
  {
    double ritzval = solver.get_eval(i);
    for (int j = 0; j < (int)n; ++j)
    {
      const double eval = op.eigenvalue(j);
      const double diff = fabs(ritzval - eval);
      const bool match = diff < tol_test;
      // std::cout << i << " " << eval << " " << ritzval << " " << "\n";
      if (match)
      {
        num_passed++;
        break;
      }
    } // j
  }   // i
  // std::cout << "num_passed " << num_passed << " num_ritzvals " <<
  // num_ritzvals << "\n";

  BOOST_TEST(num_passed == num_ritzvals);

} // BOOST_AUTO_TEST_CASE

//-----------------------------------------------------------------------------
