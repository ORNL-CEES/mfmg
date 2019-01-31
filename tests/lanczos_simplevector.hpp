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

#ifndef MFMG_LANCZOS_SIMPLEVECTOR_HPP
#define MFMG_LANCZOS_SIMPLEVECTOR_HPP

#include <cstddef>
#include <vector>

namespace mfmg
{
namespace lanczos
{

//-----------------------------------------------------------------------------
/// \brief Simple vector
///
///        Implements a simple vector with elements stored in contiguous
///        memory locations in (CPU) memory.

template <typename ScalarType_>
class SimpleVector
{

public:
  // Typedefs

  typedef ScalarType_ ScalarType;

  // Ctor/dtor

  SimpleVector(size_t dim);
  ~SimpleVector();

  // Accessors

  size_t dim() const { return _dim; }

  ScalarType &elt(size_t i);
  ScalarType const_elt(size_t i) const;

  // Operations

  void copy(const SimpleVector &x);
  void copy(const SimpleVector *x) { copy(*x); }

  void axpy(ScalarType a, const SimpleVector &x);
  void axpy(ScalarType a, const SimpleVector *x) { axpy(a, *x); }

  void scal(ScalarType a);

  ScalarType dot(const SimpleVector &x) const;
  ScalarType dot(const SimpleVector *x) const { return dot(*x); }

  ScalarType nrm2() const;

  void set_zero();

  void set_random(int seed = 0, double multiplier = 1, double cmultiplier = 0);

  void print() const;

private:
  size_t _dim;
  std::vector<ScalarType> _data;

  // Disallowed methods

  SimpleVector(const SimpleVector<ScalarType> &);
  void operator=(const SimpleVector<ScalarType> &);
};

} // namespace lanczos

} // namespace mfmg

#endif
