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

#ifndef MFMG_LANCZOS_LANCZOS_TEMPLATE_HPP
#define MFMG_LANCZOS_LANCZOS_TEMPLATE_HPP

#include <algorithm>
#include <random>
#include <vector>

#include "lanczos.hpp"
#include "lanczos_deflatedop.templates.hpp"

// This complex code has to be included before lapacke for the code to compile.
// Otherwise, it conflicts with boost or Kokkos.
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <lapacke.h>

namespace mfmg
{

/// \brief Lanczos solver: constructor
template <typename OperatorType, typename VectorType>
Lanczos<OperatorType, VectorType>::Lanczos(OperatorType const &op) : _op(op)
{
  ASSERT(_op.m() == _op.n(), "Operator must be square");
}

/// \brief Lanczos solver: perform Lanczos solve, use random initial guess
template <typename OperatorType, typename VectorType>
std::tuple<std::vector<double>, std::vector<VectorType>>
Lanczos<OperatorType, VectorType>::solve(
    boost::property_tree::ptree const &params) const
{
  const bool is_deflated = params.get<bool>("is_deflated", false);

  const int n_eigenvectors = params.get<int>("num_eigenpairs");

  int num_cycles = 1;
  int num_evecs_per_cycle = n_eigenvectors;
  if (is_deflated)
  {
    num_evecs_per_cycle = params.get<int>("num_eigenpairs_per_cycle");
    num_cycles = params.get<int>("num_cycles");
  }

  ASSERT(num_cycles >= 1, "Number of cycles must be positive");
  ASSERT(num_evecs_per_cycle >= 1,
         "Number of computed eigenpairs per cycle must be positive");

  std::vector<double> evals;
  std::vector<VectorType> evecs;

  // Form deflated operator from original operator.
  // NOTE: for regular Lanczos, it will never do any deflation
  DeflatedOperator<OperatorType, VectorType> deflated_op(_op);

  // Loop over Lanczos solves
  VectorType initial_guess(_op.n());
  for (int cycle = 0; cycle < num_cycles; ++cycle)
  {
    details_set_initial_guess(initial_guess, cycle);

    deflated_op.deflate(initial_guess);

    std::vector<double> cycle_evals;
    std::vector<VectorType> cycle_evecs;
    std::tie(cycle_evals, cycle_evecs) = details_solve_lanczos(
        deflated_op, num_evecs_per_cycle, params, initial_guess);

    // Save the eigenpairs just calculated

    // NOTE: throughout we use the term eigenpair (= eigenvector,
    // eigenvalue), though the precise terminology should be "approximate
    // eigenpairs" or "Ritz pairs."
    for (int i = 0; i < num_evecs_per_cycle; ++i)
    {
      evals.push_back(cycle_evals[i]);
      evecs.push_back(cycle_evecs[i]);
    }

    // Add eigenvectors to the set of vectors being deflated out
    // Do not deflate last cycle. For the regular Lanczos this means do not
    // deflate ever.
    if (cycle != num_cycles - 1)
    {
      deflated_op.add_deflation_vecs(cycle_evecs);
    }
  }

  // Squeeze if num_cycles * num_eigenpairs_per_cycle > n_eigenvectors
  evals.resize(n_eigenvectors);
  evecs.resize(n_eigenvectors);

  return std::make_tuple(evals, evecs);
}

/// \brief Lanczos solver: perform Lanczos solve
template <typename OperatorType, typename VectorType>
template <typename FullOperatorType>
std::tuple<std::vector<double>, std::vector<VectorType>>
Lanczos<OperatorType, VectorType>::details_solve_lanczos(
    FullOperatorType const &op, const int num_requested,
    boost::property_tree::ptree const &params, VectorType const &initial_guess)
{
  const int maxit = params.get<int>("max_iterations");
  const double tol = params.get<double>("tolerance");
  const int percent_overshoot = params.get<int>("percent_overshoot", 0);

  ASSERT(0 <= percent_overshoot && percent_overshoot < 100,
         "Lanczos overshoot percentage should be in [0, 100)");
  ASSERT(tol >= 0., "Lanczos tolerance must be non-negative");
  ASSERT(maxit >= num_requested, "Lanczos max iterations is too small to "
                                 "produce required number of eigenvectors.");

  const int n = op.n();

  std::vector<double> evals;
  std::vector<VectorType> evecs;

  // Initializations; first Lanczos vector.
  double alpha = 0;
  double beta = initial_guess.l2_norm();

  std::vector<VectorType> lanc_vectors; // Lanczos vectors

  // Create first Lanczos vector if necessary
  if (lanc_vectors.size() < 1)
    lanc_vectors.push_back(initial_guess);

  std::vector<double> t_maindiag;
  std::vector<double> t_offdiag;

  std::vector<double>
      evecs_tridiag; // eigenvectors of tridiagonal matrix, stored in flat array

  // Lanczos iteration loop
  int it = 1;
  for (int it_prev_check = 0; it <= maxit; ++it)
  {
    // Normalize lanczos vector
    ASSERT(beta, "Internal error"); // TODO: set up better check for near-zero

    lanc_vectors[it - 1] /= beta;

    if (lanc_vectors.size() < static_cast<size_t>(it + 1))
    {
      // Add new Lanczos vector
      lanc_vectors.push_back(VectorType(n));
    }

    // Apply operator.
    op.vmult(lanc_vectors[it], lanc_vectors[it - 1]);

    // Compute, apply, save Lanczos coefficients
    if (it != 1)
    {
      lanc_vectors[it].add(-beta, lanc_vectors[it - 2]);
      t_offdiag.push_back(beta);
    }

    alpha = lanc_vectors[it - 1] * lanc_vectors[it]; // = tridiag_{it,it}

    t_maindiag.push_back(alpha);

    lanc_vectors[it].add(-alpha, lanc_vectors[it - 1]);

    beta = lanc_vectors[it].l2_norm(); // = tridiag_{it+1,it}

    // Check convergence if requested
    // NOTE: an alternative here for p > 0 is
    // int((100./p)*ln(it)) > int((100./p)*ln(it-1))
    bool const first_iteration = (it == 1);
    bool const max_iterations_reached = (it == maxit);
    bool const percent_overshoot_exceeded =
        (100 * (it - it_prev_check) > percent_overshoot * it_prev_check);
    if (first_iteration || max_iterations_reached || percent_overshoot_exceeded)
    {
      const int dim_hessenberg = it;
      ASSERT((size_t)dim_hessenberg == t_maindiag.size(), "Internal error");

      // Calculate eigenpairs of tridiagonal matrix for convergence test or at
      // last iteration
      std::tie(evals, evecs_tridiag) =
          details_calc_tridiag_epairs(t_maindiag, t_offdiag, num_requested);

      if (details_check_convergence(beta, dim_hessenberg, num_requested, tol,
                                    evecs_tridiag))
      {
        break;
      }

      // Record iteration number when this check done
      it_prev_check = it;
    }
  }
  ASSERT(it >= num_requested,
         "Internal error: required number of iterations not reached");

  // Calculate full operator eigenvectors from tridiagonal eigenvectors.
  // ISSUE: may be needed to modify this code to not save all Lanczos vectors
  // but instead recalculate them for this use.
  // However this may be dangerous if the second Lanczos iteration
  // has different roundoff characteristics, e.g., due to order of
  // operation differences.
  // ISSUE: we have not taken precautions here with regard to
  // potential impacts of loss of orthogonality of Lanczos vectors.
  evecs = details_calc_evecs(num_requested, it, lanc_vectors, evecs_tridiag);

  return std::make_tuple(evals, evecs);
}

/// \brief Lanczos solver: calculate eigenpairs from tridiagonal of Lanczos
/// coefficients
template <typename OperatorType, typename VectorType>
std::tuple<std::vector<double>, std::vector<double>>
Lanczos<OperatorType, VectorType>::details_calc_tridiag_epairs(
    std::vector<double> const &t_maindiag, std::vector<double> const &t_offdiag,
    const int num_requested)
{
  const int n = t_maindiag.size();

  ASSERT(n >= 1, "Internal error: tridigonal matrix size must be positive");
  ASSERT(t_offdiag.size() == (size_t)(n - 1),
         "Internal error: mismatch in main and off-diagonal sizes");

  std::vector<double> evals;
  std::vector<double> evecs; // flat array

  if (n < num_requested)
    return std::make_tuple(evals, evecs);

  // Allocate storage
  std::vector<double> matrix(n * n, 0.);

  // Copy diagonals of the tridiagonal matrix into the full matrix
  matrix[0] = t_maindiag[0];
  for (int i = 1; i < n; ++i)
  {
    matrix[i + n * i] = t_maindiag[i];
    matrix[i - 1 + n * i] = t_offdiag[i - 1];
  }

  // Eigenvalues are returned in ascending order
  evals.resize(n);
  const lapack_int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n,
                                        matrix.data(), n, evals.data());
  ASSERT(!info, "Call to LAPACKE_dsyev failed.");

  // Save results.
  evals.resize(num_requested);
  evecs.resize(n * num_requested);
  for (int i = 0; i < num_requested; ++i)
  {
    auto first = evecs.begin() + n * i;
    auto t_first = matrix.begin() + n * i;
    auto t_last = t_first + n;
    double const norm =
        std::sqrt(std::inner_product(t_first, t_last, t_first, 0.));
    std::transform(t_first, t_last, first,
                   [norm](auto &v) { return v / norm; });
  }

  return std::make_tuple(evals, evecs);
}

/// \brief Lanczos solver: perform convergence check
template <typename OperatorType, typename VectorType>
bool Lanczos<OperatorType, VectorType>::details_check_convergence(
    double beta, const int num_evecs, const int num_requested, double tol,
    std::vector<double> const &evecs)
{
  // Must iterate at least until we have num_requested eigenpairs
  if (num_evecs < num_requested)
    return false;

  bool is_converged = true;

  // Terminate if every approximage eigenvalue has converged to tolerance
  // ISSUE: here ignoring possible nuances regarding the correctness
  // of this check.
  // NOTE: k may be desirable to "scale" the stopping criterion
  // based on (estimate of) matrix norm or similar.
  for (int i = 0; i < num_requested; ++i)
  {
    const double bound = beta * std::abs(evecs[num_evecs - 1 + num_evecs * i]);
    is_converged = is_converged && bound <= tol;
  }

  return is_converged;
}

/// \brief Lanczos solver: calculate full (approx) eigenvectors from tridiag
/// eigenvectors
template <typename OperatorType, typename VectorType>
std::vector<VectorType> Lanczos<OperatorType, VectorType>::details_calc_evecs(
    const int num_requested, const int n,
    std::vector<VectorType> const &lanc_vectors,
    std::vector<double> const &evecs_tridiag)
{
  auto dim = lanc_vectors[0].size();

  std::vector<VectorType> evecs(num_requested, VectorType(dim));

  // Matrix-matrix product to convert tridiagonal eigenvectors to operator
  // eigenvectors
  for (int i = 0; i < num_requested; ++i)
  {
    evecs[i] = 0.0;
    for (int j = 0; j < n; ++j)
      evecs[i].add(evecs_tridiag[j + n * i], lanc_vectors[j]);
  }

  return evecs;
}

template <typename OperatorType, typename VectorType>
void Lanczos<OperatorType, VectorType>::details_set_initial_guess(
    VectorType &initial_guess, int seed)
{
  // Set initial guess to constant vector to try to capture "smooth" eigenmodes
  // of PDE problems
  initial_guess = 1.;

  // Add random noise to the guess
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0, 1);
  std::transform(initial_guess.begin(), initial_guess.end(),
                 initial_guess.begin(), [&](auto &v) { return v + dist(gen); });
}

} // namespace mfmg

#endif
