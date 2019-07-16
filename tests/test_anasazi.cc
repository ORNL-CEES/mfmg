/*************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#define BOOST_TEST_MODULE anasazi

#include <mfmg/dealii/anasazi.templates.hpp>

#include <boost/test/data/test_case.hpp>

#include <cmath>
#include <cstdio>

#include "lanczos_simpleop.templates.hpp"
#include "main.cc"

namespace bdata = boost::unit_test::data;
namespace tt = boost::test_tools;

namespace Anasazi
{

template <typename VectorType>
class OperatorTraits<double, mfmg::MultiVector<VectorType>,
                     mfmg::SimpleOperator<VectorType>>
{
  using MultiVectorType = mfmg::MultiVector<VectorType>;
  using OperatorType = mfmg::SimpleOperator<VectorType>;

public:
  static void Apply(const OperatorType &op, const MultiVectorType &x,
                    MultiVectorType &y)
  {
    auto n_vectors = x.n_vectors();

    ASSERT(x.size() == y.size(), "");
    ASSERT(y.n_vectors() == n_vectors, "");

    for (int i = 0; i < n_vectors; i++)
      op.vmult(*y[i], *x[i]);
  }
};

} // namespace Anasazi

BOOST_DATA_TEST_CASE(anasazi,
                     bdata::make({1, 2}) * bdata::make({1, 2, 3, 5, 10}) *
                         bdata::make({false, true}),
                     multiplicity, n_distinct_eigenvalues,
                     multiple_initial_guesses)
{
  using namespace mfmg;

  using VectorType = dealii::Vector<double>;
  using OperatorType = SimpleOperator<VectorType>;

  int const n = 1000;
  int const n_eigenvectors = n_distinct_eigenvalues * multiplicity;
  int const n_initial_guess = multiple_initial_guesses ? n_eigenvectors : 1;

  OperatorType op(n, multiplicity);

  boost::property_tree::ptree anasazi_params;
  anasazi_params.put("number of eigenvectors", n_eigenvectors);

  anasazi_params.put("max_iterations", 1000);
  anasazi_params.put("tolerance", 1e-2);

  mfmg::AnasaziSolver<OperatorType, VectorType> solver(op);

  VectorType initial_guess_vector(n);
  initial_guess_vector = 1.;

  // Add random noise to the guess
  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(0, 1);
  std::transform(initial_guess_vector.begin(), initial_guess_vector.end(),
                 initial_guess_vector.begin(),
                 [&](auto &v) { return v + dist(gen); });

  std::vector<std::shared_ptr<VectorType>> initial_guess(n_initial_guess);
  for (int i = 0; i < n_initial_guess; ++i)
  {
    initial_guess[i] = std::make_shared<VectorType>(initial_guess_vector);
  }
  std::vector<double> computed_evals;
  std::vector<VectorType> computed_evecs;
  std::tie(computed_evals, computed_evecs) =
      solver.solve(anasazi_params, initial_guess);

  auto ref_evals = op.get_evals();

  BOOST_TEST(computed_evals.size() == n_eigenvectors);

  // Loop to ensure each Ritz value is near an eigenvalue.
  const double tolerance =
      anasazi_params.get<double>("tolerance"); // this may need adjustment

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
