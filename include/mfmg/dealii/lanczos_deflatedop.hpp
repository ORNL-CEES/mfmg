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

#ifndef MFMG_LANCZOS_DEFLATEDOP_HPP
#define MFMG_LANCZOS_DEFLATEDOP_HPP

#include <mfmg/common/exceptions.hpp>
#include <mfmg/common/operator.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace mfmg
{

//-----------------------------------------------------------------------------
/// \brief Deflated operator
///
/// Given an non-deflated operator, a new operator is constructed
/// with a subspace represented by a set of vectors projected out.

template <typename OperatorType, typename VectorType>
class DeflatedOperator
{
public:
  DeflatedOperator(OperatorType const &op);

  DeflatedOperator(DeflatedOperator<OperatorType, VectorType> const &) = delete;
  DeflatedOperator<OperatorType, VectorType> &
  operator=(DeflatedOperator<OperatorType, VectorType> const &) = delete;

  // Operations
  void vmult(VectorType &y, VectorType const &x) const;

  size_t m() const { return _base_op.m(); }
  size_t n() const { return _base_op.n(); }

  void add_deflation_vecs(std::vector<VectorType> const &vecs);

  void deflate(VectorType &vec) const;

private:
  OperatorType const &_base_op; // reference to the base operator object
  std::vector<VectorType> _deflation_vecs; // vectors to deflate out
};

} // namespace mfmg

#endif
