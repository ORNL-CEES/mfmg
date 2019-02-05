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
#include <cmath>

#include "lanczos_deflatedop.hpp"

namespace mfmg
{

//-----------------------------------------------------------------------------
/// \brief Deflated operator: constructor

template <typename VectorType>
DeflatedOperator<VectorType>::DeflatedOperator(
    const Operator<VectorType> &base_op)
    : _base_op(base_op)
{
}

//-----------------------------------------------------------------------------
/// \brief Deflated operator: destructor

template <typename BaseOperatorType>
DeflatedOperator<BaseOperatorType>::~DeflatedOperator()
{

  for (int i = 0; i < _deflation_vecs.size(); ++i)
  {
    delete _deflation_vecs[i];
  }
}

//-----------------------------------------------------------------------------
/// \brief Deflated operator: apply operator to a vector

template <typename VectorType>
void DeflatedOperator<VectorType>::apply(VectorType const &x, VectorType &y,
                                         OperatorMode mode) const
{
  // NOTE: to save a vec, we will assume the initial guess is already deflated.
  // NOTE: the deflation needs to be applied as part of the operator,
  // not just the initial guess, because the deflation vecs are not
  // exact eigenvecs, and because roundoff behaviors can cause the deflated
  // evecs to grow back otherwise.
  // ISSUE: it is not required to deflate at every step;
  // a future modification may take this into account.  There are
  // some methods for determining when / how often needed.
  _base_op.apply(x, y, mode);

  deflate(y);
}

//-----------------------------------------------------------------------------
/// \brief Deflated operator: add more vectors to the set of deflation vectors

template <typename VectorType>
void DeflatedOperator<VectorType>::add_deflation_vecs(Vectors_t vecs)
{

  const int num_old = _deflation_vecs.size();
  const int num_new = vecs.size();
  const int num_total = num_old + num_new;

  // ISSUE: we are using a very unsophisticated modified Gram Schmidt
  // with permutation to essentially perform a rank revealing QR
  // factorization.  Much more sophisticated and robust methods exist.
  // ISSUE: this has no BLAS-3 performance characteristics.
  // This is bad for performance but possibly useful for
  // portability, since only BLAS-1 operations are required.

  // Copy in new vectors

  for (int i = 0; i < num_new; ++i)
    _deflation_vecs.push_back(new VectorType(*vecs[i]));

  // Orthogonalize new vectors with respect to old vectors.

  for (int i = 0; i < num_new; ++i)
  {
    for (int j = 0; j < num_old; ++j)
    {
      ScalarType a = (*_deflation_vecs[j]) * (*_deflation_vecs[num_old + i]);
      _deflation_vecs[num_old + i]->add(-a, (*_deflation_vecs[j]));
    }
  }

  // Orthonormalize new vectors with respect to each other.

  std::vector<int> ind(num_new);
  for (int i = 0; i < num_new; ++i)
  {
    ind[i] = num_old + i;
  }

  for (int i = 0; i < num_new; ++i)
  {

    // Find longest vector of those left.

    double dot_best = -1;

    for (int j = i; j < num_new; ++j)
    {
      const double dot_this =
          (double)((*_deflation_vecs[ind[j]]) * (*_deflation_vecs[ind[j]]));

      if (dot_this > dot_best)
      {
        dot_best = dot_this;
        int tmp = ind[i];
        ind[i] = ind[j];
        ind[j] = tmp;
      }
    }

    // Normalize.

    // ISSUE: we are not accounting for possible rank deficiency here.

    double norm = std::sqrt(dot_best);
    assert(norm != (double)0.); // ISSUE need better test for near-zero here.
    (*_deflation_vecs[ind[i]]) *= (ScalarType)(1 / norm);

    // Orthogonalize all later vectors against this one.

    for (int j = i + 1; j < num_new; ++j)
    {
      ScalarType a = (*_deflation_vecs[ind[i]]) * (*_deflation_vecs[ind[j]]);
      _deflation_vecs[ind[j]]->add(-a, *_deflation_vecs[ind[i]]);
    }

  } // i
}

//-----------------------------------------------------------------------------
/// \brief Deflated operator: apply the deflation (projection) to a vector

template <typename VectorType>
void DeflatedOperator<VectorType>::deflate(VectorType &vec) const
{

  // Apply (I - VV^T) to a vector.

  for (int i = 0; i < _deflation_vecs.size(); ++i)
  {
    ScalarType a = (*_deflation_vecs[i]) * vec;
    vec.add(-a, *_deflation_vecs[i]);
  }
}

} // namespace mfmg

#endif
