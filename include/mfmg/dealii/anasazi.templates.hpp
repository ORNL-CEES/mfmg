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

#include "anasazi.hpp"
#include "anasazi_traits.hpp"
#include "belos_traits.hpp"
#include <AnasaziBasicEigenproblem.hpp>
#include <AnasaziFactory.hpp>

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

  Anasazi::BasicEigenproblem<double, MultiVectorType, OperatorType> problem;

  MultiVectorType mv_initial_guess(1);
  *mv_initial_guess[0] = initial_guess;

  // Indicate the symmetry of the problem to allow wider range of solvers (to
  // include LOBPCG)
  problem.setHermitian(true);
  problem.setA(Teuchos::rcpFromRef(_op));
  problem.setNEV(n_eigenvectors);
  problem.setInitVec(Teuchos::rcpFromRef(mv_initial_guess));

  bool r = problem.setProblem();
  ASSERT(r, "Anasazi could not setup the problem");

  Teuchos::ParameterList solverParams;
  solverParams.set("Convergence Tolerance", params.get<double>("tolerance"));
  solverParams.set("Maximum Iterations", params.get<int>("max_iterations"));
  solverParams.set("Which", "SM");
  // Specify that the residuals norms should not be scaled by their eigenvalues
  // for the purposing of deciding convergence
  solverParams.set("Relative Convergence Tolerance", false);
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
