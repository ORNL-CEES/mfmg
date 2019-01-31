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
Lanczos<OperatorType>::Lanczos(const OperatorType &op, int num_requested,
                               int maxit, double tol,
                               unsigned int percent_overshoot,
                               unsigned int verbosity)
    : _op(op)
      //, _lanc_vectors()
      ,
      _num_requested(num_requested), _maxit(maxit), _tol(tol),
      _percent_overshoot(percent_overshoot), _verbosity(verbosity),
      _dim(op.dim())
{
  assert(this->_num_requested >= 1);
  assert(this->_maxit >= 0);
  assert(this->_maxit >= this->_num_requested &&
         "maxit too small to produce required number of eigenvectors.");
  assert(this->_tol >= 0.);
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

  for (int i = 0; i < _lanc_vectors.size(); ++i)
  {
    delete _lanc_vectors[i];
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

  typename OperatorType::VectorType guess(_dim);
  guess.set_random();
  solve(guess);
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: perform Lanczos solve

template <typename OperatorType>
void Lanczos<OperatorType>::solve(
    const typename OperatorType::VectorType &guess)
{

  // Initializations; first Lanczos vector.

  ScalarType alpha = 0;
  ScalarType beta = guess.nrm2();

  if (_lanc_vectors.size() < 1)
  {
    // Create first Lanczos vector.
    _lanc_vectors.push_back(new VectorType(_dim));
  }
  // Set to initial guess; later normalize.
  _lanc_vectors[0]->copy(guess);

  Scalars_t t_maindiag;
  Scalars_t t_offdiag;

  // Lanczos iteration loop.

  for (int it = 1, it_prev_check = 0; it <= _maxit; ++it)
  {

    // Normalize lanczos vector.

    assert(beta != 0); // TODO: set up better check for near-zero.
    _lanc_vectors[it - 1]->scal(1 / beta);

    if (_lanc_vectors.size() < it + 1)
    {
      // Add new Lanczos vector
      _lanc_vectors.push_back(new VectorType(_dim));
    }

    // Apply operator.

    _op.apply(*_lanc_vectors[it], *_lanc_vectors[it - 1]);

    // Compute, apply, save lanczos coefficients.

    if (1 != it)
    {
      _lanc_vectors[it]->axpy(-beta, _lanc_vectors[it - 2]);
      t_offdiag.push_back(beta);
    } // if

    alpha = _lanc_vectors[it - 1]->dot(_lanc_vectors[it]); // = tridiag_{it,it}

    t_maindiag.push_back(alpha);

    _lanc_vectors[it]->axpy(-alpha, _lanc_vectors[it - 1]);

    beta = _lanc_vectors[it]->nrm2(); // = tridiag_{it+1,it}

    // Check convergence if requested.

    const bool do_check =
        1 == it || _maxit == it ||
        100 * (it - it_prev_check) > _percent_overshoot * it_prev_check;
    // NOTE: an alternative here for p > 0 is
    // int((100./p)*ln(it)) > int((100./p)*ln(it-1))

    if (do_check)
    {
      // Calc eigenpairs of tridiag matrix for conv test or at last it.
      calc_tridiag_epairs_(it, t_maindiag, t_offdiag);

      if (check_convergence_(beta))
      {
        break;
      }

      // Record iteration number when this check done.
      it_prev_check = it;
    }

  } // for it

  // Calc full operator evecs from tridiag evecs.
  // ISSUE: may be needed to modify this code to not save all lanc vecs
  // but instead recalculate them for this use.
  // However this may be dangerous if the second lanczos iteration
  // has different roundoff characteristics, e.g., due to order of
  // operatioon differences.
  // ISSUE: we have not taken precautions here with regard to
  // potential impacts of loss of orthogonality of lanczos vectors.

  calc_evecs_();

  if (_verbosity > 0)
  {
    std::cout << std::endl;
  }
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: calculate eigenpairs from tridiag of lanczos coeffs

template <typename OperatorType>
void Lanczos<OperatorType>::calc_tridiag_epairs_(int it, Scalars_t &t_maindiag,
                                                 Scalars_t &t_offdiag)
{
  assert(it >= 1);
  assert(it <= this->_maxit);
  assert(t_maindiag.size() == it);
  assert(t_offdiag.size() == it - 1);

  _dim_tridiag = it;
  const int k = _dim_tridiag;

  // Set up matrix

  // TODO: an implementation of this with fewerpointers and more encapsulation.

  auto matrix = new double[k * k];

  for (int i = 0; i < k * k; ++i)
  {
    matrix[i] = 0;
  }

  // Copy tridiag matrix to full matrix.

  matrix[0] = (double)t_maindiag[0];
  for (int i = 1; i < k; ++i)
  {
    matrix[i + k * i] = (double)t_maindiag[i];
    matrix[i + k * (i - 1)] = (double)t_offdiag[i - 1];
    matrix[i - 1 + k * i] = (double)t_offdiag[i - 1];
  }

  // LAPACK eigenvalue/vector solve.

  // NOTE: this part can be replaced if desired with some platform-specific
  // library.
  // NOTE: for accuracy we are using double here rather than ScalarType;
  // may not be needed.
  // ISSUE: LAPACK has other solvers that might be more efficient here.

  auto t_evals_r = new double[k];
  auto t_evals_i = new double[k];
  auto t_evecs = new double[k * k];

  // Do all in double, regardless of ScalarType (for now)
  const lapack_int info =
      LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', k, matrix, k, t_evals_r,
                    t_evals_i, NULL, k, t_evecs, k);

  // Sort into ascending order

  std::vector<int> sort_index(k);
  for (int i = 0; i < k; ++i)
  {
    sort_index[i] = i;
  }

  std::sort(sort_index.begin(), sort_index.end(),
            [&](int i, int j) { return t_evals_r[i] < t_evals_r[j]; });

  // Output requested eigenvalues.

  if (_verbosity > 0)
  {
    std::cout.width(4);
    std::cout << "It " << k;
    std::cout.precision(4);
    std::cout << "   evals ";
    for (int i = 0; i < k && i < _num_requested; ++i)
    {
      std::cout << " " << std::fixed << t_evals_r[sort_index[i]];
    }
    if (_dim_tridiag < _num_requested)
    {
      std::cout << "\n";
    }
  }

  // Save results.

  if (k >= _num_requested)
  {
    _evals.resize(_num_requested);
    _evecs_tridiag.resize(k * _num_requested);

    for (int i = 0; i < _num_requested; ++i)
    {
      const int si = sort_index[i];
      _evals[i] = (ScalarType)t_evals_r[si];
      double sum = 0;
      for (int j = 0; j < k; ++j)
      {
        sum += t_evecs[j + k * si] * t_evecs[j + k * si];
      }
      const double norm = sqrt(sum);
      for (int j = 0; j < k; ++j)
      {
        _evecs_tridiag[j + k * i] = (ScalarType)(t_evecs[j + k * si] / norm);
      }
    } // for i
  }   // if

  // Terminate

  delete matrix;
  delete t_evals_r;
  delete t_evals_i;
  delete t_evecs;

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
bool Lanczos<OperatorType>::check_convergence_(
    typename OperatorType::ScalarType beta)
{

  // Must iterate at least until we have _num_requested eigenpairs.
  if (_dim_tridiag < _num_requested)
  {
    return false;
  }

  const int n = _dim_tridiag;

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
  // NOTE: it may be desirable to "scale" the stopping criterion
  // based on (estimate of) matrix norm or similar.
  for (int i = 0; i < _num_requested; ++i)
  {
    const double bound =
        (double)beta * abs((double)_evecs_tridiag[n - 1 + n * i]);
    is_converged = is_converged && bound <= _tol;
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
void Lanczos<OperatorType>::calc_evecs_()
{
  assert(_evecs.size() == 0);
  assert(_dim_tridiag >= _num_requested);

  const int k = _dim_tridiag;

  // Matrix-matrix product to convert tridiag evecs to operator evecs.

  for (int i = 0; i < _num_requested; ++i)
  {
    _evecs.push_back(new VectorType(_dim));
    _evecs[i]->set_zero();

    for (int j = 0; j < _dim_tridiag; ++j)
    {
      _evecs[i]->axpy(_evecs_tridiag[j + k * i], _lanc_vectors[j]);
    }
    // _evecs[i]->print();
  }
}

} // namespace lanczos

} // namespace mfmg

#endif
