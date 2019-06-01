/**************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                            *
 * All rights reserved.                                                   *
 *                                                                        *
 * This file is part of the mfmg library. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the  *
 * top-level directory                                                    *
 *                                                                        *
 * SPDX-License-Identifier: BSD-3-Clause                                  *
 **************************************************************************/

#define BOOST_TEST_MODULE lanczos_device

#include <mfmg/common/lanczos.templates.hpp>
#include <mfmg/cuda/utils.cuh>

#include <boost/test/data/test_case.hpp>

#include <cmath>
#include <cstdio>

#include "lanczos_simpleop.templates.hpp"
#include "main.cc"

namespace bdata = boost::unit_test::data;
namespace tt = boost::test_tools;

BOOST_DATA_TEST_CASE(lanczos,
                     bdata::make({1, 2}) * bdata::make({false, true}) *
                         bdata::make({1, 2, 3, 5}),
                     multiplicity, is_deflated, n_distinct_eigenvalues)
{
  using namespace mfmg;

  using VectorType =
      dealii::LinearAlgebra::distributed::Vector<double,
                                                 dealii::MemorySpace::CUDA>;
  using OperatorType = SimpleOperator<VectorType>;

  int const n = 1000;
  int const n_eigenvectors = n_distinct_eigenvalues * multiplicity;

  OperatorType op(n, multiplicity);

  boost::property_tree::ptree lanczos_params;
  lanczos_params.put("is_deflated", is_deflated);
  lanczos_params.put("num_eigenpairs", n_eigenvectors);
  if (is_deflated)
  {
    lanczos_params.put("num_cycles", multiplicity);
    lanczos_params.put("num_eigenpairs_per_cycle", n_distinct_eigenvalues);
  }
  else
  {
    if (multiplicity > 1)
    {
      // Skip test when using regular Lanczos for multiplicities larger than 1
      return;
    }
  }

  lanczos_params.put("max_iterations", 200);
  lanczos_params.put("tolerance", 1e-2);
  lanczos_params.put("percent_overshoot", 5);

  Lanczos<OperatorType, VectorType> solver(op);

  dealii::LinearAlgebra::distributed::Vector<double> initial_guess_host(n);
  initial_guess_host = 1.;

  // Add random noise to the guess
  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(0, 1);
  std::transform(initial_guess_host.begin(), initial_guess_host.end(),
                 initial_guess_host.begin(),
                 [&](auto &v) { return v + dist(gen); });
  auto initial_guess_dev = copy_from_host(initial_guess_host);

  std::vector<double> computed_evals;
  std::vector<VectorType> computed_evecs;
  std::tie(computed_evals, computed_evecs) =
      solver.solve(lanczos_params, initial_guess_dev);

  auto ref_evals = op.get_evals();

  BOOST_TEST(computed_evals.size() == n_eigenvectors);

  // Loop to ensure each Ritz value is near an eigenvalue.
  double const tolerance =
      lanczos_params.get<double>("tolerance"); // this may need adjustment

  std::sort(ref_evals.begin(), ref_evals.end());
  std::sort(computed_evals.begin(), computed_evals.end());

  for (int i = 0; i < n_eigenvectors; i++)
    BOOST_TEST(computed_evals[i] == ref_evals[i], tt::tolerance(tolerance));

  // Testing eigenvectors is tricky. Specifically, when multiplicity > 1, one
  // gets a subspace of possible solutions. One way to test that is to
  //   a) test that each eigenvector is indeed an eigenvector corresponding to
  //   the eigenvalue
  //   b) test that the eigenvectors corresponding to the same eigenvalue are
  //   orthogonal
  // In addition, from the numerical perspective, one should also care about
  // scaling/normalization things. It is also unclear what tolerance should be
  // used here.  For now, we are just happy to have something here.
  for (int i = 0; i < n_eigenvectors; i++)
  {
    VectorType result(n);
    op.vmult(result, computed_evecs[i]);
    result.add(-computed_evals[i], computed_evecs[i]);
    BOOST_TEST(result.l2_norm() < tolerance);
  }
}
