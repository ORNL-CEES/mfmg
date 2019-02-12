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

#include <boost/test/data/test_case.hpp>

#include <cmath>
#include <cstdio>

#include "lanczos_simpleop.templates.hpp"
#include "main.cc"

namespace bdata = boost::unit_test::data;
namespace tt = boost::test_tools;

BOOST_DATA_TEST_CASE(lanczos, bdata::make({1, 2}) * bdata::make({false, true}),
                     multiplicity, is_deflated)
{
  using namespace mfmg;

  using VectorType = dealii::Vector<double>;
  using OperatorType = SimpleOperator<VectorType>;

  const int n = 1000;
  const int n_distinct_eigenvalues = 3;
  const int n_eigenvectors = n_distinct_eigenvalues * multiplicity;

  OperatorType op(n, multiplicity);

  boost::property_tree::ptree lanczos_params;
  lanczos_params.put("is_deflated", is_deflated);
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
    lanczos_params.put("num_eigenpairs", n_eigenvectors);
  }
  lanczos_params.put("max_iterations", 200);
  lanczos_params.put("tolerance", 1e-2);
  lanczos_params.put("percent_overshoot", 5);

  Lanczos<OperatorType, VectorType> solver(op, lanczos_params);

  solver.solve();

  auto ref_evals = op.get_evals();
  auto computed_evals = solver.get_evals();

  BOOST_TEST(computed_evals.size() == n_eigenvectors);

  // Loop to ensure each Ritz value is near an eigenvalue.
  // TODO: make a more complete check of correctness of result
  const double tolerance =
      lanczos_params.get<double>("tolerance"); // this may need adjustment

  std::sort(ref_evals.begin(), ref_evals.end());
  std::sort(computed_evals.begin(), computed_evals.end());

  for (int i = 0; i < n_eigenvectors; i++)
    BOOST_TEST(computed_evals[i] == ref_evals[i], tt::tolerance(tolerance));
}
