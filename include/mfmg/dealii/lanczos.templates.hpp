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
#include <cassert>
#include <iostream>
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
Lanczos<OperatorType, VectorType>::Lanczos(
    OperatorType const &op, boost::property_tree::ptree const &params)
    : _op(op)
{
  _is_deflated = params.get<bool>("is_deflated");
  _maxit = params.get<int>("max_iterations");
  _tol = params.get<double>("tolerance");
  _percent_overshoot = params.get<int>("percent_overshoot", 0);
  _verbosity = params.get<unsigned int>("verbosity", 0);

  assert(0 <= _percent_overshoot && _percent_overshoot <= 100);

  if (_is_deflated)
  {
    _num_evecs_per_cycle = params.get<int>("num_eigenpairs_per_cycle");
    _num_cycles = params.get<int>("num_cycles");

    assert(_num_evecs_per_cycle >= 1);
    assert(_num_cycles >= 1);
    assert(_maxit >= _num_evecs_per_cycle &&
           "maxit too small to produce required number of eigenvectors.");
  }
  else
  {
    _num_requested = params.get<int>("num_eigenpairs");

    assert(_num_requested >= 1);
    assert(_maxit >= _num_requested &&
           "maxit too small to produce required number of eigenvectors.");
  }

  assert(_op.m() == _op.n() && "Operator must be square");

  assert(_maxit >= 0);
  assert(_tol >= 0.);
}

/// \brief Lanczos solver: accessor for (approximate) eigenvalue
template <typename OperatorType, typename VectorType>
std::vector<double> const &Lanczos<OperatorType, VectorType>::get_evals() const
{
  return _evals;
}

template <typename OperatorType, typename VectorType>
double Lanczos<OperatorType, VectorType>::get_eval(int i) const
{
  assert(i >= 0 && (size_t)i < _evals.size());
  return _evals[i];
}

/// \brief Lanczos solver: accessor for (approximate) eigenvector
template <typename OperatorType, typename VectorType>
VectorType const &Lanczos<OperatorType, VectorType>::get_evec(int i) const
{
  assert(i >= 0 && (size_t)i < _evecs.size());
  return _evecs[i];
}

/// \brief Lanczos solver: accessor for (approximate) eigenvectors
template <typename OperatorType, typename VectorType>
std::vector<VectorType> const &
Lanczos<OperatorType, VectorType>::get_evecs() const
{
  return _evecs;
}

/// \brief Lanczos solver: perform Lanczos solve, use random initial guess
template <typename OperatorType, typename VectorType>
void Lanczos<OperatorType, VectorType>::solve()
{
  if (!_is_deflated)
  {
    // By default set initial guess to a random vector.
    VectorType initial_guess(_op.n());
    {
      std::mt19937 gen(0);
      std::uniform_real_distribution<double> dist(0, 1);
      std::generate(initial_guess.begin(), initial_guess.end(),
                    [&]() { return 1 + dist(gen); });
    }
    details_solve_lanczos(_op, _num_requested, initial_guess, _evals, _evecs);
  }
  else
  {
    // Form deflated operator from original operator.
    DeflatedOperator<OperatorType, VectorType> deflated_op(_op);

    // Loop over Lanczos solves
    for (int cycle = 0; cycle < _num_cycles; ++cycle)
    {

      if (_verbosity > 0)
      {
        std::cout << "----------------------------------------"
                     "---------------------------------------"
                  << std::endl;
        std::cout << "Lanczos solve " << cycle + 1 << ":" << std::endl;
      }

      // Perform Lanczos solve, initial guess is a linear combination of a
      // constant vector (to try to capture "smooth" eigenmodes of PDE problems)
      // and a random vector based on different random seeds.
      // ISSUE: should a different initial guess strategy be used?
      VectorType initial_guess(_op.n());
      {
        std::mt19937 gen(cycle);
        std::uniform_real_distribution<double> dist(0, 1);
        std::generate(initial_guess.begin(), initial_guess.end(),
                      [&]() { return 1 + dist(gen); });
      }

      // Deflate initial guess
      deflated_op.deflate(initial_guess);

      std::vector<double> evals;
      std::vector<VectorType> evecs;
      details_solve_lanczos(deflated_op, _num_evecs_per_cycle, initial_guess,
                            evals, evecs);

      // Save the eigenpairs just calculated

      // NOTE: throughout we use the term eigenpair (= eigenvector,
      // eigenvalue), though the precise terminology should be "approximate
      // eigenpairs" or "Ritz pairs."
      for (int i = 0; i < _num_evecs_per_cycle; ++i)
      {
        _evals.push_back(evals[i]);
        _evecs.push_back(evecs[i]);
      }

      // Add eigenvectors to the set of vectors being deflated out
      if (cycle != _num_cycles - 1)
      {
        deflated_op.add_deflation_vecs(evecs);
      }
    }
  }
}

/// \brief Lanczos solver: perform Lanczos solve
template <typename OperatorType, typename VectorType>
template <typename FullOperatorType>
void Lanczos<OperatorType, VectorType>::details_solve_lanczos(
    FullOperatorType const &op, const int num_requested,
    VectorType const &initial_guess, std::vector<double> &evals,
    std::vector<VectorType> &evecs)
{
  const int n = op.n();

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
  for (int it_prev_check = 0; it <= _maxit; ++it)
  {
    // Normalize lanczos vector
    assert(beta != 0); // TODO: set up better check for near-zero
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
    const bool do_check =
        it == 1 || it == _maxit ||
        (100 * (it - it_prev_check) > _percent_overshoot * it_prev_check);
    if (do_check)
    {
      // Calculate eigenpairs of tridiagonal matrix for convvergence test or at
      // last iteration
      details_calc_tridiag_epairs(t_maindiag, t_offdiag, num_requested, evals,
                                  evecs_tridiag);

      if (details_check_convergence(beta, num_requested, _tol, evecs_tridiag))
      {
        break;
      }

      // Record iteration number when this check done
      it_prev_check = it;
    }
  }
  assert(it >= num_requested);

  // Calculate full operator eigenvectors from tridiagonal eigenvectors.
  // ISSUE: may be needed to modify this code to not save all Lanczos vectors
  // but instead recalculate them for this use.
  // However this may be dangerous if the second Lanczos iteration
  // has different roundoff characteristics, e.g., due to order of
  // operation differences.
  // ISSUE: we have not taken precautions here with regard to
  // potential impacts of loss of orthogonality of Lanczos vectors.
  details_calc_evecs(num_requested, it, lanc_vectors, evecs_tridiag, evecs);

  if (_verbosity > 0)
    std::cout << std::endl;
}

/// \brief Lanczos solver: calculate eigenpairs from tridiagonal of Lanczos
/// coefficients
template <typename OperatorType, typename VectorType>
void Lanczos<OperatorType, VectorType>::details_calc_tridiag_epairs(
    std::vector<double> const &t_maindiag, std::vector<double> const &t_offdiag,
    const int num_requested, std::vector<double> &evals,
    std::vector<double> &evecs)
{
  const int n = t_maindiag.size();

  assert(n >= 1);
  assert(t_offdiag.size() == (size_t)(n - 1));

  // Allocate storage
  std::vector<double> matrix(n * n, 0.);
  std::vector<double> t_evals_r(n, 0.);
  std::vector<double> t_evals_i(n, 0.);
  std::vector<double> t_evecs(n * n, 0.);

  // Copy diagonals of the tridiagonal matrix into the full matrix
  matrix[0] = t_maindiag[0];
  for (int i = 1; i < n; ++i)
  {
    matrix[i + n * i] = t_maindiag[i];
    matrix[i + n * (i - 1)] = t_offdiag[i - 1];
    matrix[i - 1 + n * i] = t_offdiag[i - 1];
  }

  // LAPACK eigenvalue/vector solve.

  // NOTE: this part can be replaced if desired with some platform-specific
  // library.
  // NOTE: for accuracy we are using double here rather than double;
  // may not be needed.
  // ISSUE: LAPACK has other solvers that might be more efficient here.

  // Do all in double, regardless of double (for now)
  const lapack_int info = LAPACKE_dgeev(
      LAPACK_COL_MAJOR, 'N', 'V', n, matrix.data(), n, t_evals_r.data(),
      t_evals_i.data(), NULL, n, t_evecs.data(), n);
  ASSERT(!info, "Call to LAPACKE_dgeev failed.");

  // Compute permutation for ascending order
  std::vector<int> perm_index(n);
  std::iota(perm_index.begin(), perm_index.end(), 0);
  std::sort(perm_index.begin(), perm_index.end(),
            [&](int i, int j) { return t_evals_r[i] < t_evals_r[j]; });

  // Output requested eigenvalues
  if (_verbosity > 0)
  {
    std::cout.width(4);
    std::cout << "It " << n;
    std::cout.precision(4);
    std::cout << "   evals ";
    for (int i = 0; i < n && i < num_requested; ++i)
    {
      std::cout << " " << std::fixed << t_evals_r[perm_index[i]];
    }
    if (n < num_requested)
    {
      std::cout << "\n";
    }
  }

  // Save results.
  // FIXME: this does not follow details style, it should do a single thing
  // unconditionally
  if (n >= num_requested)
  {
    evals.resize(num_requested);
    evecs.resize(n * num_requested);

    for (int i = 0; i < num_requested; ++i)
    {
      const int si = perm_index[i];

      evals[i] = t_evals_r[si];

      auto first = evecs.begin() + n * i;
      auto t_first = t_evecs.begin() + n * si;
      auto t_last = t_first + n;
      double const norm =
          std::sqrt(std::inner_product(t_first, t_last, t_first, 0.));
      std::transform(t_first, t_last, first,
                     [norm](auto &v) { return v / norm; });
    }
  }
}

/// \brief Lanczos solver: perform convergence check
template <typename OperatorType, typename VectorType>
bool Lanczos<OperatorType, VectorType>::details_check_convergence(
    double beta, const int num_requested, double tol,
    std::vector<double> const &evecs)
{
  const int n = evecs.size();

  // Must iterate at least until we have num_requested eigenpairs
  if (n < num_requested)
  {
    return false;
  }

  bool is_converged = true;

  if (_verbosity > 0)
  {
    std::cout.precision(4);
    std::cout << std::fixed;
    // std::cout << "   " << beta;
    std::cout << "   bounds ";
  }

  // Terminate if every approximage eigenvalue has converged to tolerance
  // ISSUE: here ignoring possible nuances regarding the correctness
  // of this check.
  // NOTE: k may be desirable to "scale" the stopping criterion
  // based on (estimate of) matrix norm or similar.
  for (int i = 0; i < num_requested; ++i)
  {
    const double bound = beta * abs(evecs[n - 1 + n * i]);
    is_converged = is_converged && bound <= tol;
    if (_verbosity > 0)
    {
      std::cout << " " << bound;
    }
  }

  if (_verbosity > 0)
  {
    std::cout << "\n";
  }

  return is_converged;
}

/// \brief Lanczos solver: calculate full (approx) eigenvectors from tridiag
/// eigenvectors
template <typename OperatorType, typename VectorType>
void Lanczos<OperatorType, VectorType>::details_calc_evecs(
    const int num_requested, const int n,
    std::vector<VectorType> const &lanc_vectors,
    std::vector<double> const &evecs_tridiag, std::vector<VectorType> &evecs)
{
  assert(evecs.empty());

  auto dim = lanc_vectors[0].size();

  evecs.resize(num_requested, VectorType(dim));

  // Matrix-matrix product to convert tridiagonal eigenvectors to operator
  // eigenvectors
  for (int i = 0; i < num_requested; ++i)
  {
    evecs[i] = 0.0;
    for (int j = 0; j < n; ++j)
      evecs[i].add(evecs_tridiag[j + n * i], lanc_vectors[j]);
  }
}

} // namespace mfmg

#endif
