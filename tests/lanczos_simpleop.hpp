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

#ifndef MFMG_LANCZOS_SIMPLEOP_HPP
#define MFMG_LANCZOS_SIMPLEOP_HPP

#include <cstddef>
#include <vector>

namespace mfmg
{
namespace lanczos
{

//-----------------------------------------------------------------------------
/// \brief Simple test operator
///
///        A diagonal matrix with equally spaced eigenvalues of some
///        multiplicity.

template <typename VectorType_>
class SimpleOp
{

public:
  // Typedefs

  typedef VectorType_ VectorType;
  typedef typename VectorType::value_type ScalarType;
  typedef typename std::vector<VectorType *> Vectors_t;

  // Ctor/dtor

  SimpleOp(size_t dim, size_t multiplicity = 1);
  ~SimpleOp();

  // Accessors

  size_t dim() const { return _dim; }

  ScalarType eigenvalue(size_t i) const { return diag_value_(i); }

  // Operations

  void apply(VectorType const &vin, VectorType &vout) const;

private:
  size_t _dim;
  size_t _multiplicity;

  // This will be a diagonal matrix; specify the diag entries here

  ScalarType diag_value_(size_t i) const
  {
    return (ScalarType)(1 + i / _multiplicity);
  }

  // Disallowed methods

  SimpleOp(const SimpleOp<VectorType> &);
  void operator=(const SimpleOp<VectorType> &);
};

//-----------------------------------------------------------------------------

} // namespace lanczos

} // namespace mfmg

#endif
