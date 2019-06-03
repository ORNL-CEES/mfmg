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

#ifndef MFMG_ANASAZI_TEMPLATES_HPP
#define MFMG_ANASAZI_TEMPLATES_HPP

#include <mfmg/common/exceptions.hpp>
#include <mfmg/common/operator.hpp>
#include <mfmg/dealii/anasazi.hpp>
#include <mfmg/dealii/anasazi_traits.hpp>
#ifdef HAVE_ANASAZI_BELOS
#include <mfmg/dealii/belos_traits.hpp>
#endif
#include <AnasaziBasicEigenproblem.hpp>
#include <AnasaziFactory.hpp>

namespace mfmg
{

template <typename VectorType>
class DiagonalOperator : public OperatorBase<VectorType>
{
public:
  using operator_type = DiagonalOperator<VectorType>;
  using vector_type = dealii::Vector<double>;

  DiagonalOperator(std::vector<double> const &diag)
  {
    _diag.resize(diag.size());
    std::transform(diag.begin(), diag.end(), _diag.begin(),
                   [](auto v) { return (v ? 1. / v : 0.); });
  }

  DiagonalOperator(const DiagonalOperator<VectorType> &) = delete;
  DiagonalOperator<VectorType> &
  operator=(const DiagonalOperator<VectorType> &) = delete;

  void vmult(VectorType &y, VectorType const &x) const
  {
    auto dim = _diag.size();
    ASSERT(x.size() == dim, "");
    ASSERT(y.size() == dim, "");

    for (size_t i = 0; i < dim; ++i)
      y[i] = _diag[i] * x[i];
  }

  size_t m() const override { return _diag.size(); }
  size_t n() const override { return _diag.size(); }

private:
  std::vector<double> _diag;
};
} // namespace mfmg

namespace Anasazi
{

template <typename VectorType>
class OperatorTraits<double, mfmg::MultiVector<VectorType>,
                     mfmg::OperatorBase<VectorType>>
{
  using MultiVectorType = mfmg::MultiVector<VectorType>;
  using OperatorType = mfmg::OperatorBase<VectorType>;

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

namespace mfmg
{

/// \brief Anasazi solver: constructor
template <typename OperatorType, typename VectorType>
AnasaziSolver<OperatorType, VectorType>::AnasaziSolver(OperatorType const &op)
    : _op(op)
{
  ASSERT(_op.m() == _op.n(), "Operator must be square");
}

/// \brief Anasazi solver: perform Anasazi solve, use random initial guess
template <typename OperatorType, typename VectorType>
std::tuple<std::vector<double>, std::vector<VectorType>>
AnasaziSolver<OperatorType, VectorType>::solve(
    boost::property_tree::ptree const &params, VectorType initial_guess) const
{
  using MultiVectorType = MultiVector<VectorType>;
  const int n_eigenvectors = params.get<int>("num_eigenpairs");

  Anasazi::BasicEigenproblem<double, MultiVectorType,
                             mfmg::OperatorBase<VectorType>>
      problem;

  MultiVectorType mv_initial_guess(1);
  *mv_initial_guess[0] = initial_guess;

  // Indicate the symmetry of the problem to allow wider range of solvers (to
  // include LOBPCG)
  problem.setHermitian(true);
  problem.setA(Teuchos::rcpFromRef(_op));
  problem.setNEV(n_eigenvectors);
  problem.setInitVec(Teuchos::rcpFromRef(mv_initial_guess));

  DiagonalOperator<VectorType> prec(_op.get_diag_elements());
  if (params.get<bool>("use_preconditioner", true))
    problem.setPrec(Teuchos::rcpFromRef(prec));

  bool r = problem.setProblem();
  ASSERT(r, "Anasazi could not setup the problem");

  Teuchos::ParameterList solverParams;
  solverParams.set("Convergence Tolerance", params.get<double>("tolerance"));
  solverParams.set("Maximum Iterations", params.get<int>("max_iterations"));
  solverParams.set("Which", "SM");
  // Specify that the residuals norms should not be scaled by their eigenvalues
  // for the purposing of deciding convergence
  solverParams.set("Relative Convergence Tolerance", false);
  // Specity whether solver should employ a full orthogonalization technique
  // This is may solve Chebyshev failure for large number of eigenvectors as
  // noted in "Basis selection in LOBPCG" 2006 paper by Hetmaniuk and Lehoucq
  solverParams.set("Full Ortho", params.get<bool>("full_ortho", true));
  // solverParams.set("Verbosity", 127);
  auto solver = Anasazi::Factory::create("LOBPCG", Teuchos::rcpFromRef(problem),
                                         solverParams);

  Anasazi::ReturnType rr = solver->solve();
  ASSERT(rr == Anasazi::Converged, "Anasazi could not solve the problem");

  // Extract solution
  Anasazi::Eigensolution<double, MultiVectorType> solution =
      solver->getProblem().getSolution();

  int a_n_eigenvectors = solution.numVecs;
  std::vector<Anasazi::Value<double>> &a_eigenvalues = solution.Evals;
  std::vector<int> &a_index = solution.index;

  size_t num_converged = std::min(a_n_eigenvectors, n_eigenvectors);

  std::vector<double> evals(num_converged);
  std::vector<VectorType> evecs(num_converged);

  for (size_t i = 0; i < a_eigenvalues.size(); i++)
  {
    ASSERT(a_index[i] == 0, "Encountered complex eigenvalue");
    evals[i] = a_eigenvalues[i].realpart;
    evecs[i] = *((*solution.Evecs)[i]);
  }

  return std::make_tuple(evals, evecs);
}

} // namespace mfmg

#endif
