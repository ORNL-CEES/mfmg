/**************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                            *
 * All rights reserved.                                                   *
 *                                                                        *
 * This file is part of the mfmg library. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the  *
 * top-level directory                                                    *
 *                                                                        *
 * SPDX-License-Identifier: BSD-3-Clause                                  *
 *************************************************************************/

#ifndef MFMG_LANCZOS_DEFLATEDOP_TEMPLATE_HPP
#define MFMG_LANCZOS_DEFLATEDOP_TEMPLATE_HPP

#include <cmath>

#include "lanczos_deflatedop.hpp"

namespace mfmg
{

/// \brief Deflated operator: constructor
template <typename OperatorType, typename VectorType>
DeflatedOperator<OperatorType, VectorType>::DeflatedOperator(
    const OperatorType &base_op)
    : _base_op(base_op)
{
}

/// \brief Deflated operator: apply operator to a vector
template <typename OperatorType, typename VectorType>
void DeflatedOperator<OperatorType, VectorType>::vmult(
    VectorType &y, VectorType const &x) const
{
  // NOTE: to save a vec, we will assume the initial guess is already deflated.
  // NOTE: the deflation needs to be applied as part of the operator,
  // not just the initial guess, because the deflation vecs are not
  // exact eigenvecs, and because roundoff behaviors can cause the deflated
  // evecs to grow back otherwise.
  // ISSUE: it is not required to deflate at every step;
  // a future modification may take this into account.  There are
  // some methods for determining when / how often needed.
  _base_op.vmult(y, x);

  deflate(y);
}

/// \brief Deflated operator: add more vectors to the set of deflation vectors
///
/// ISSUE: we are using a very unsophisticated modified Gram Schmidt
/// with permutation to essentially perform a rank revealing QR
/// factorization. Much more sophisticated and robust methods exist.
///
/// ISSUE: this has no BLAS-3 performance characteristics.
/// This is bad for performance but possibly useful for
/// portability, since only BLAS-1 operations are required.
template <typename OperatorType, typename VectorType>
void DeflatedOperator<OperatorType, VectorType>::add_deflation_vecs(
    std::vector<VectorType> const &vecs)
{

  const int num_old = _deflation_vecs.size();
  const int num_new = vecs.size();

  // Copy in new vectors
  for (auto const &v : vecs)
    _deflation_vecs.push_back(v);

  // These have to be computed *after* push_back as it may invalidate iterators
  auto vold_start = _deflation_vecs.begin();
  auto vold_end = vold_start + num_old;
  auto vnew_start = vold_end;
  auto vnew_end = vnew_start + num_new;

  // Orthogonalize new vectors with respect to old vectors
  std::for_each(vnew_start, vnew_end, [&](auto &new_v) {
    std::for_each(vold_start, vold_end,
                  [&new_v](auto const &v) { new_v.add(-(new_v * v), v); });
  });

  // Orthonormalize new vectors with respect to each other (with an additional
  // twist of doing that in decreasing norm order)
  std::vector<int> perm_ind(num_new);
  std::iota(perm_ind.begin(), perm_ind.end(), num_old);
  for (int i = 0; i < num_new; ++i)
  {
    // Find longest vector of those left
    double dot_best = -1;

    for (int j = i; j < num_new; ++j)
    {
      const double dot_this =
          _deflation_vecs[perm_ind[j]] * _deflation_vecs[perm_ind[j]];

      if (dot_this > dot_best)
      {
        dot_best = dot_this;
        std::swap(perm_ind[i], perm_ind[j]);
      }
    }

    // Normalize.

    // FIXME: we are not accounting for possible rank deficiency here
    double norm = std::sqrt(dot_best);
    ASSERT(norm, "Internal error: zero norm"); // FIXME need better test for
                                               // near-zero here.
    _deflation_vecs[perm_ind[i]] /= norm;

    // Orthogonalize all later vectors against this one
    for (int j = i + 1; j < num_new; ++j)
    {
      double a = _deflation_vecs[perm_ind[i]] * _deflation_vecs[perm_ind[j]];
      _deflation_vecs[perm_ind[j]].add(-a, _deflation_vecs[perm_ind[i]]);
    }
  }
}

/// \brief Deflated operator: apply the deflation (projection) to a vector
template <typename OperatorType, typename VectorType>
void DeflatedOperator<OperatorType, VectorType>::deflate(VectorType &vec) const
{
  // Apply (I - VV^T) to a vector
  std::for_each(_deflation_vecs.begin(), _deflation_vecs.end(),
                [&vec](auto const &dvec) { vec.add(-(vec * dvec), dvec); });
}

} // namespace mfmg

#endif
