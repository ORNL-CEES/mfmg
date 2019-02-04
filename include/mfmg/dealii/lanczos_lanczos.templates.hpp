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
#include <vector>

#include "cblas.h"
#include "lanczos_lanczos.hpp"

// This complex code has to be included before lapacke for the code to compile.
// Otherwise, it conflicts with boost or Kokkos.
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include "lapacke.h"

namespace mfmg
{
namespace lanczos
{

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: constructor

template <typename OperatorType>
Lanczos<OperatorType>::Lanczos(OperatorType const &op,
                               boost::property_tree::ptree const &params)
    : _op(op)
{
  _num_requested = params.get<int>("num_eigenpairs");
  _maxit = params.get<int>("max_iterations");
  _tol = params.get<double>("tolerance");
  _percent_overshoot = params.get<int>("percent_overshoot", 0);
  _verbosity = params.get<unsigned int>("verbosity", 0);

  assert(_num_requested >= 1);
  assert(_maxit >= 0);
  assert(_maxit >= _num_requested &&
         "maxit too small to produce required number of eigenvectors.");
  assert(_tol >= 0.);
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: destructor

template <typename OperatorType>
Lanczos<OperatorType>::~Lanczos()
{

  for (int i = 0; i < _evecs.size(); ++i)
  {
    delete _evecs[i];
  }
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: accessor for (approximate) eigenvalue

template <typename OperatorType>
typename OperatorType::ScalarType Lanczos<OperatorType>::get_eval(int i) const
{
  assert(i >= 0);
  assert(i < _evals.size());

  return _evals[i];
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: accessor for (approximate) eigenvector

template <typename OperatorType>
typename OperatorType::VectorType *Lanczos<OperatorType>::get_evec(int i) const
{
  assert(i >= 0);
  assert(i < _evecs.size());

  // ISSUE: giving users pointer to internal data that shouldn't be modified.
  return _evecs[i];
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: accessor for (approximate) eigenvectors

template <typename OperatorType>
typename OperatorType::Vectors_t Lanczos<OperatorType>::get_evecs() const
{

  // ISSUE: giving users pointer to internal data that shouldn't be modified.
  return _evecs;
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: perform Lanczos solve, use random initial guess

template <typename OperatorType>
void Lanczos<OperatorType>::solve()
{

  // By default set initial guess to a random vector.

  typename OperatorType::VectorType initial_guess(_op.dim());
  initial_guess.set_random();
  details_solve_lanczos(_num_requested, initial_guess, _evals, _evecs);
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: perform Lanczos solve

template <typename OperatorType>
void Lanczos<OperatorType>::details_solve_lanczos(
    const int num_requested, VectorType const &initial_guess, Scalars_t &evals,
    Vectors_t &evecs)
{
  // Initializations; first Lanczos vector.

  ScalarType alpha = 0;
  ScalarType beta = initial_guess.nrm2();

  Vectors_t lanc_vectors; // Lanczos vectors

  if (lanc_vectors.size() < 1)
  {
    // Create first Lanczos vector.
    lanc_vectors.push_back(new VectorType(_op.dim()));
  }
  // Set to initial guess; later normalize.
  lanc_vectors[0]->copy(initial_guess);

  Scalars_t t_maindiag;
  Scalars_t t_offdiag;

  Scalars_t evecs_tridiag; // eigenvecs of tridiag matrix, stored in flat array

  // Lanczos iteration loop.
  int it = 1;
  for (int it_prev_check = 0; it <= _maxit; ++it)
  {
    // Normalize lanczos vector.
    assert(beta != 0); // TODO: set up better check for near-zero.
    lanc_vectors[it - 1]->scal(1 / beta);

    if (lanc_vectors.size() < it + 1)
    {
      // Add new Lanczos vector
      lanc_vectors.push_back(new VectorType(_op.dim()));
    }

    // Apply operator.

    _op.apply(*lanc_vectors[it - 1], *lanc_vectors[it]);

    // Compute, apply, save lanczos coefficients.

    if (1 != it)
    {
      lanc_vectors[it]->axpy(-beta, lanc_vectors[it - 2]);
      t_offdiag.push_back(beta);
    } // if

    alpha = lanc_vectors[it - 1]->dot(lanc_vectors[it]); // = tridiag_{it,it}

    t_maindiag.push_back(alpha);

    lanc_vectors[it]->axpy(-alpha, lanc_vectors[it - 1]);

    beta = lanc_vectors[it]->nrm2(); // = tridiag_{it+1,it}

    // Check convergence if requested.

    const bool do_check =
        1 == it || _maxit == it ||
        100 * (it - it_prev_check) > _percent_overshoot * it_prev_check;
    // NOTE: an alternative here for p > 0 is
    // int((100./p)*ln(it)) > int((100./p)*ln(it-1))

    if (do_check)
    {
      // Calc eigenpairs of tridiag matrix for conv test or at last it.
      details_calc_tridiag_epairs(t_maindiag, t_offdiag, num_requested, evals,
                                  evecs_tridiag);

      if (details_check_convergence(beta, num_requested, _tol, evecs_tridiag))
      {
        break;
      }

      // Record iteration number when this check done.
      it_prev_check = it;
    }
  }
  assert(it >= num_requested);

  // Calc full operator evecs from tridiag evecs.
  // ISSUE: may be needed to modify this code to not save all lanc vecs
  // but instead recalculate them for this use.
  // However this may be dangerous if the second lanczos iteration
  // has different roundoff characteristics, e.g., due to order of
  // operatioon differences.
  // ISSUE: we have not taken precautions here with regard to
  // potential impacts of loss of orthogonality of lanczos vectors.
  details_calc_evecs(num_requested, it, lanc_vectors, evecs_tridiag, evecs);

  if (_verbosity > 0)
  {
    std::cout << std::endl;
  }

  for (int i = 0; i < lanc_vectors.size(); ++i)
  {
    delete lanc_vectors[i];
  }
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: calculate eigenpairs from tridiag of lanczos coeffs

template <typename OperatorType>
void Lanczos<OperatorType>::details_calc_tridiag_epairs(
    Scalars_t const &t_maindiag, Scalars_t const &t_offdiag,
    const int num_requested, Scalars_t &evals, Scalars_t &evecs)
{
  const int n = t_maindiag.size();

  assert(n >= 1);
  assert(t_offdiag.size() == n - 1);

  // Allocate storage
  std::vector<double> matrix(n * n, 0.);
  std::vector<double> t_evals_r(n, 0.);
  std::vector<double> t_evals_i(n, 0.);
  std::vector<double> t_evecs(n * n, 0.);

  // Copy diagonals of the tridiagonal matrix into the full matrix
  matrix[0] = (double)t_maindiag[0];
  for (int i = 1; i < n; ++i)
  {
    matrix[i + n * i] = (double)t_maindiag[i];
    matrix[i + n * (i - 1)] = (double)t_offdiag[i - 1];
    matrix[i - 1 + n * i] = (double)t_offdiag[i - 1];
  }

  // LAPACK eigenvalue/vector solve.

  // NOTE: this part can be replaced if desired with some platform-specific
  // library.
  // NOTE: for accuracy we are using double here rather than ScalarType;
  // may not be needed.
  // ISSUE: LAPACK has other solvers that might be more efficient here.

  // Do all in double, regardless of ScalarType (for now)
  const lapack_int info = LAPACKE_dgeev(
      LAPACK_COL_MAJOR, 'N', 'V', n, matrix.data(), n, t_evals_r.data(),
      t_evals_i.data(), NULL, n, t_evecs.data(), n);

  // Sort into ascending order

  std::vector<int> sort_index(n);
  for (int i = 0; i < n; ++i)
  {
    sort_index[i] = i;
  }

  std::sort(sort_index.begin(), sort_index.end(),
            [&](int i, int j) { return t_evals_r[i] < t_evals_r[j]; });

  // Output requested eigenvalues.

  if (_verbosity > 0)
  {
    std::cout.width(4);
    std::cout << "It " << n;
    std::cout.precision(4);
    std::cout << "   evals ";
    for (int i = 0; i < n && i < num_requested; ++i)
    {
      std::cout << " " << std::fixed << t_evals_r[sort_index[i]];
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
      const int si = sort_index[i];
      evals[i] = static_cast<ScalarType>(t_evals_r[si]);
      double sum = 0;
      for (int j = 0; j < n; ++j)
      {
        sum += t_evecs[j + n * si] * t_evecs[j + n * si];
      }
      const double norm = sqrt(sum);
      for (int j = 0; j < n; ++j)
      {
        evecs[j + n * i] = static_cast<ScalarType>(t_evecs[j + n * si] / norm);
      }
    }
  }

#if 0

void LAPACK_dgeev( char* jobvl, char* jobvr, lapack_int* n, double* a,
                   lapack_int* lda, double* wr, double* wi, double* vl,
                   lapack_int* ldvl, double* vr, lapack_int* ldvr, double* work,
                   lapack_int* lwork, lapack_int *info );

lapack_int LAPACKE_dgeev( int matrix_order, char jobvl, char jobvr,
                          lapack_int n, double* a, lapack_int lda, double* wr,
                          double* wi, double* vl, lapack_int ldvl, double* vr,
                          lapack_int ldvr );

lapack_int LAPACKE_dgeev_work( int matrix_order, char jobvl, char jobvr,
                               lapack_int n, double* a, lapack_int lda,
                               double* wr, double* wi, double* vl,
                               lapack_int ldvl, double* vr, lapack_int ldvr,
                               double* work, lapack_int lwork );


http://www.netlib.org/lapack/explore-html/d9/d8e/group__double_g_eeigen_ga66e19253344358f5dee1e60502b9e96f.html#ga66e19253344358f5dee1e60502b9e96f
#endif
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: perform convergence check

template <typename OperatorType>
bool Lanczos<OperatorType>::details_check_convergence(ScalarType beta,
                                                      const int num_requested,
                                                      double tol,
                                                      Scalars_t const &evecs)
{
  const int n = evecs.size();

  // Must iterate at least until we have num_requested eigenpairs.
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

  // Terminate if every approx eval has converged to tolerance
  // ISSUE: here ignoring possible nuances regarding the correctness
  // of this check.
  // NOTE: k may be desirable to "scale" the stopping criterion
  // based on (estimate of) matrix norm or similar.
  for (int i = 0; i < num_requested; ++i)
  {
    const double bound = (double)beta * abs((double)evecs[n - 1 + n * i]);
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

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: calculate full (approx) evecs from tridiag evecs
template <typename OperatorType>
void Lanczos<OperatorType>::details_calc_evecs(const int num_requested,
                                               const int n,
                                               Vectors_t const &lanc_vectors,
                                               Scalars_t const &evecs_tridiag,
                                               Vectors_t &evecs)
{
  assert(evecs.size() == 0);

  auto dim = lanc_vectors[0]->dim();

  // Matrix-matrix product to convert tridiag evecs to operator evecs.
  for (int i = 0; i < num_requested; ++i)
  {
    evecs.push_back(new VectorType(dim));
    evecs[i]->set_zero();

    for (int j = 0; j < n; ++j)
    {
      evecs[i]->axpy(evecs_tridiag[j + n * i], lanc_vectors[j]);
    }
    // _evecs[i]->print();
  }
}

} // namespace lanczos

} // namespace mfmg

#endif
