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

#ifndef MFMG_LANCZOS_DEFLATEDOP_HPP
#define MFMG_LANCZOS_DEFLATEDOP_HPP

#include <cstddef>
#include <vector>

namespace mfmg
{
namespace lanczos
{

//-----------------------------------------------------------------------------
/// \brief Deflated operator
///
///        Given an undeflated operator, a new operator is constructed
///        with a subspace represented by a set of vectors projected out.

template <typename BaseOperatorType_>
class DeflatedOp
{

public:
  // Typedefs

  typedef BaseOperatorType_ BaseOperatorType;
  typedef typename BaseOperatorType_::VectorType VectorType;
  typedef typename VectorType::ScalarType ScalarType;
  typedef typename std::vector<VectorType *> Vectors_t;

  // Ctor/dtor

  DeflatedOp(const BaseOperatorType &base_op);
  ~DeflatedOp();

  // Accessors

  size_t dim() const { return _dim; } // operator and vector dimension

  // Operations

  void apply(VectorType const &vin, VectorType &vout) const;

  void add_deflation_vecs(Vectors_t vecs);

  void deflate(VectorType &vec) const;

private:
  const BaseOperatorType_ &_base_op; // reference to the base operator object

  size_t _dim; // operator and vector dimension

  Vectors_t _deflation_vecs; // vectors to deflate out

  // Disallowed methods

  DeflatedOp(const DeflatedOp<BaseOperatorType> &);
  void operator=(const DeflatedOp<BaseOperatorType> &);
};

} // namespace lanczos

} // namespace mfmg

#endif
