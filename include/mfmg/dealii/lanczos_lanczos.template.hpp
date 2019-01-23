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

#ifndef _LANCZOS_LANCZOS_TEMPLATE_HPP_
#define _LANCZOS_LANCZOS_TEMPLATE_HPP_

#include <cassert>
#include <algorithm>
#include <vector>
#include <iostream>

#include "cblas.h"
#include "lapacke.h"

#include "lanczos_lanczos.hpp"

namespace mfmg::lanczos
{

//-----------------------------------------------------------------------------

template<typename Op_t>
Lanczos<Op_t>::Lanczos(const Op_t& op, int num_requested, int maxit, double tol,
                       unsigned int percent_overshoot, unsigned int verbosity)
  : op_(op)
  //, lanc_vectors_()
  , num_requested_(num_requested)
  , maxit_(maxit)
  , tol_(tol)
  , percent_overshoot_(percent_overshoot)
  , verbosity_(verbosity)
  , dim_(op.dim()) {
  assert(this->num_requested_ >= 1);
  assert(this->maxit_ >= 0);
  assert(this->maxit_ >= this->num_requested_ &&
         "maxit too small to produce required number of eigenvectors.");
  assert(this->tol_ >= 0.);
}

//-----------------------------------------------------------------------------

template<typename Op_t>
Lanczos<Op_t>::~Lanczos() {

  for (int i = 0; i < evecs_.size(); ++i) {
    delete evecs_[i];
  }

  for (int i = 0; i < lanc_vectors_.size(); ++i) {
    delete lanc_vectors_[i];
  }
}

//-----------------------------------------------------------------------------

template<typename Op_t>
typename Op_t::Scalar_t Lanczos<Op_t>::get_eval(int i) const {
  assert(i >= 0);
  assert(i < evals_.size());

  return evals_[i];
}

//-----------------------------------------------------------------------------

template<typename Op_t>
typename Op_t::Vector_t* Lanczos<Op_t>::get_evec(int i) const {
  assert(i >= 0);
  assert(i < evecs_.size());

  // ISSUE: giving users pointer to internal data that shouldn't be modified.
  return evecs_[i];
}

//-----------------------------------------------------------------------------

template<typename Op_t>
typename Op_t::Vectors_t Lanczos<Op_t>::get_evecs() const {

  // ISSUE: giving users pointer to internal data that shouldn't be modified.
  return evecs_;
}

//-----------------------------------------------------------------------------

template<typename Op_t>
void Lanczos<Op_t>::solve() {

  // By default set initial guess to a random vector.

  typename Op_t::Vector_t guess(dim_);
  guess.set_random();
  solve(guess);
}

//-----------------------------------------------------------------------------

template<typename Op_t>
void Lanczos<Op_t>::solve(const typename Op_t::Vector_t& guess) {

  // Initializations; first Lanczos vector.

  Scalar_t alpha = 0;
  Scalar_t beta = guess.nrm2();

  if (lanc_vectors_.size() < 1) {
    // Create first Lanczos vector.
    lanc_vectors_.push_back(new Vector_t(dim_));
  }
  // Set to initial guess; later normalize.
  lanc_vectors_[0]->copy(guess);

  Scalars_t t_maindiag;
  Scalars_t t_offdiag;

  // Lanczos iteration loop.

  for (int it=1, it_prev_check=0 ; it<=maxit_; ++it) {

    // Normalize lanczos vector.

    assert(beta != 0); // TODO: set up better check for near-zero.
    lanc_vectors_[it-1]->scal(1 / beta);

    if (lanc_vectors_.size() < it+1) {
      // Add new Lanczos vector
      lanc_vectors_.push_back(new Vector_t(dim_));
    }

    // Apply operator.

    op_.apply(*lanc_vectors_[it], *lanc_vectors_[it-1]);

    // Compute, apply, save lanczos coefficients.

    if (1 != it) {
      lanc_vectors_[it]->axpy(-beta, lanc_vectors_[it-2]);
      t_offdiag.push_back(beta);
    } // if

    alpha = lanc_vectors_[it-1]->dot(lanc_vectors_[it]); // = tridiag_{it,it}

    t_maindiag.push_back(alpha);

    lanc_vectors_[it]->axpy(-alpha, lanc_vectors_[it-1]);

    beta = lanc_vectors_[it]->nrm2(); // = tridiag_{it+1,it}

    // Check convergence if requested.

    const bool do_check = 1 == it || maxit_ == it ||
      100 * (it - it_prev_check) > percent_overshoot_ * it_prev_check;
      // NOTE: an alternative here for p > 0 is
      // int((100./p)*ln(it)) > int((100./p)*ln(it-1))

    if (do_check) {
      // Calc eigenpairs of tridiag matrix for conv test or at last it.
      calc_tridiag_epairs_(it, t_maindiag, t_offdiag);

      if (check_convergence_(beta)) {
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

  if (verbosity_ > 0) {
    std::cout << std::endl;
  }
}

//-----------------------------------------------------------------------------

template<typename Op_t>
void Lanczos<Op_t>::calc_tridiag_epairs_(int it, Scalars_t& t_maindiag,
                                         Scalars_t& t_offdiag) {
  assert(it >= 1);
  assert(it <= this->maxit_);
  assert(t_maindiag.size() == it);
  assert(t_offdiag.size() == it-1);

  dim_tridiag_ = it;
  const int k = dim_tridiag_;

  // Set up matrix

  // TODO: an implementation of this with fewerpointers and more encapsulation.

  auto matrix = new double[k*k];

  for (int i=0; i<k*k; ++i) {
    matrix[i] = 0;
  }

  // Copy tridiag matrix to full matrix.

  matrix[0] = (double)t_maindiag[0];
  for (int i=1; i<k; ++i) {
    matrix[i + k*i] = (double)t_maindiag[i];
    matrix[i + k*(i-1)] = (double)t_offdiag[i-1];
    matrix[i-1 + k*i] = (double)t_offdiag[i-1];
  }

  // LAPACK eigenvalue/vector solve.

  // NOTE: this part can be replaced if desired with some platform-specific
  // library.
  // NOTE: for accuracy we are using double here rather than Scalar_t;
  // may not be needed.
  // ISSUE: LAPACK has other solvers that might be more efficient here.

  auto t_evals_r = new double[k];
  auto t_evals_i = new double[k];
  auto t_evecs = new double[k*k];

  // Do all in double, regardless of Scalar_t (for now)
  const lapack_int info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V',
      k, matrix, k, t_evals_r, t_evals_i, NULL, k, t_evecs, k);

  // Sort into ascending order

  std::vector<int> sort_index(k);
  for (int i=0; i<k; ++i) {
    sort_index[i] = i;
  }

  std::sort(sort_index.begin(), sort_index.end(),
    [&](int i, int j) { return t_evals_r[i] <
                               t_evals_r[j]; });

  // Output requested eigenvalues.

  if (verbosity_ > 0) {
    std::cout.width(4);
    std::cout << "It " << k;
    std::cout.precision(4);
    std::cout << "   evals ";
    for (int i=0; i<k && i<num_requested_; ++i) {
      std::cout << " " << std::fixed << t_evals_r[sort_index[i]];
    }
    if (dim_tridiag_ < num_requested_) {
      std::cout << "\n";
    }
  }

  // Save results.

  if (k >= num_requested_) {
    evals_.resize(num_requested_);
    evecs_tridiag_.resize(k * num_requested_);

    for (int i=0; i<num_requested_; ++i) {
      const int si = sort_index[i];
      evals_[i] = (Scalar_t)t_evals_r[si];
      double sum = 0;
      for (int j=0; j<k; ++j) {
        sum += t_evecs[j+k*si] * t_evecs[j+k*si];
      }
      const double norm = sqrt(sum);
      for (int j=0; j<k; ++j) {
        evecs_tridiag_[j+k*i] = (Scalar_t)(t_evecs[j+k*si] / norm);
      }
    } // for i
  } // if

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

template<typename Op_t>
bool Lanczos<Op_t>::check_convergence_(typename Op_t::Scalar_t beta) {

  // Must iterate at least until we have num_requested_ eigenpairs.
  if (dim_tridiag_ < num_requested_) {
    return false;
  }

  const int n = dim_tridiag_;

  bool is_converged = true;

  if (verbosity_ > 0) {
    std::cout.precision(4);
    std::cout << std::fixed;
    //std::cout << "   " << beta;
    std::cout << "   bounds ";
  }

  // Terminate if every approx eval has converged to tolerance
  // ISSUE: here ignoring possible nuances regarding the correctness
  // of this check.
  // NOTE: it may be desirable to "scale" the stopping criterion
  // based on (estimate of) matrix norm or similar.
  for (int i=0; i<num_requested_; ++i) {
    const double bound = (double)beta * abs((double)evecs_tridiag_[n-1+n*i]);
    is_converged = is_converged && bound <= tol_;
    if (verbosity_ > 0) {
      std::cout << " " << bound;
    }
  }

  if (verbosity_ > 0) {
    std::cout << "\n";
  }

  return is_converged;
}

//-----------------------------------------------------------------------------

template<typename Op_t>
void Lanczos<Op_t>::calc_evecs_() {
  assert(evecs_.size() == 0);
  assert(dim_tridiag_ >= num_requested_);

  const int k = dim_tridiag_;

  // Matrix-matrix product to convert tridiag evecs to operator evecs.

  for (int i=0; i<num_requested_; ++i) {
    evecs_.push_back(new Vector_t(dim_));
    evecs_[i]->set_zero();

    for (int j=0; j<dim_tridiag_; ++j) {
      evecs_[i]->axpy(evecs_tridiag_[j+k*i], lanc_vectors_[j]);
    }
    //evecs_[i]->print();
  }
}

//-----------------------------------------------------------------------------

} // namespace mfmg::lanczos

#endif // _LANCZOS_LANCZOS_TEMPLATE_HPP_

//=============================================================================
