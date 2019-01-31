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

#ifndef MFMG_LANCZOS_DEFLATEDOP_TEMPLATE_HPP
#define MFMG_LANCZOS_DEFLATEDOP_TEMPLATE_HPP

#include <cassert>

#include "lanczos_deflatedop.hpp"

namespace mfmg::lanczos
{

//-----------------------------------------------------------------------------
/// \brief Deflated operator: constructor

template<typename BaseOp_t>
DeflatedOp<BaseOp_t>::DeflatedOp(const BaseOp_t& base_op)
  : base_op_(base_op)
  , dim_(base_op.dim()) {
}

//-----------------------------------------------------------------------------
/// \brief Deflated operator: destructor

template<typename BaseOp_t>
DeflatedOp<BaseOp_t>::~DeflatedOp() {

  for (int i=0; i<deflation_vecs_.size(); ++i) {
    delete deflation_vecs_[i];
  }
}

//-----------------------------------------------------------------------------
/// \brief Deflated operator: apply operator to a vector

template<typename BaseOp_t>
void DeflatedOp<BaseOp_t>::apply(Vector_t& vout, const Vector_t& vin) const {

  // NOTE: to save a vec, we will assume the initial guess is already deflated.
  // NOTE: the deflation needs to be applied as part of the operator,
  // not just the initial guess, because the deflation vecs are not
  // exact eigenvecs, and because roundoff behaviors can cause the deflated
  // evecs to grow back otherwise.
  // ISSUE: it is not required to deflate at every step;
  // a future modification may take this into account.  There are
  // some methods for determining when / how often needed.

  base_op_.apply(vout, vin);

  deflate(vout);
}

//-----------------------------------------------------------------------------
/// \brief Deflated operator: add more vectors to the set of deflation vectors

template<typename BaseOp_t>
void DeflatedOp<BaseOp_t>::add_deflation_vecs(Vectors_t vecs) {

  const int num_old = deflation_vecs_.size();
  const int num_new = vecs.size();
  const int num_total = num_old + num_new;

  // ISSUE: we are using a very unsophisticated modified Gram Schmidt
  // with permutation to essentially perform a rank revealing QR
  // factorization.  Much more sophisticated and robust methods exist.
  // ISSUE: this has no BLAS-3 performance characteristics.
  // This is bad for performance but possibly useful for
  // portability, since only BLAS-1 operations are required.

  // Copy in new vectors

  for (int i=0; i<num_new; ++i) {
    deflation_vecs_.push_back(new Vector_t(dim_));
    deflation_vecs_.back()->copy(vecs[i]);
  }

  // Orthogonalize new vectors with respect to old vectors.

  for (int i=0; i<num_new; ++i) {
    for (int j=0; j<num_old; ++j) {
      Scalar_t a = deflation_vecs_[j]->dot(deflation_vecs_[num_old+i]);
      deflation_vecs_[num_old+i]->axpy(-a, deflation_vecs_[j]);
    }
  }

  // Orthonormalize new vectors with respect to each other.

  std::vector<int> ind(num_new);
  for (int i=0; i<num_new; ++i) {
    ind[i] = num_old + i;
  }

  for (int i=0; i<num_new; ++i) {

    // Find longest vector of those left.

    double dot_best = -1;

    for (int j=i; j<num_new; ++j) {
      const double dot_this = (double) deflation_vecs_[ind[j]]->dot(
                                       deflation_vecs_[ind[j]]);

      if (dot_this > dot_best) {
        dot_best = dot_this;
        int tmp = ind[i];
        ind[i] = ind[j];
        ind[j] = tmp;
      }

    }

    // Normalize.

    // ISSUE: we are not accounting for possible rank deficiency here.

    double norm = sqrt(dot_best);
    assert(norm != (double)0.); // ISSUE need better test for near-zero here.
    deflation_vecs_[ind[i]]->scal((Scalar_t)(1/norm));

    // Orthogonalize all later vectors against this one.

    for (int j=i+1; j<num_new; ++j) {
      Scalar_t a = deflation_vecs_[ind[i]]->dot(
                   deflation_vecs_[ind[j]]);
      deflation_vecs_[ind[j]]->axpy(-a, deflation_vecs_[ind[i]]);
    }

  } // i
}

//-----------------------------------------------------------------------------
/// \brief Deflated operator: apply the deflation (projection) to a vector

template<typename BaseOp_t>
void DeflatedOp<BaseOp_t>::deflate(Vector_t& vec) const {

  // Apply (I - VV^T) to a vector.

  for (int i=0; i<deflation_vecs_.size(); ++i) {
    Scalar_t a = deflation_vecs_[i]->dot(vec);
    vec.axpy(-a, deflation_vecs_[i]);
  }
}

//-----------------------------------------------------------------------------

} // namespace mfmg::lanczos

#endif // _LANCZOS_DEFLATEDOP_TEMPLATE_HPP_

//=============================================================================
