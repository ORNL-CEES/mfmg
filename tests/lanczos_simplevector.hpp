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

template <typename Number>
class SimpleVector
{

public:
  typedef Number value_type;
  typedef SimpleVector<Number> VectorType;

  // Ctor/dtor

  SimpleVector(const size_t n);
  ~SimpleVector();

  // Accessors
  size_t size() const { return _dim; }

  Number &operator[](size_t i);
  Number operator[](size_t i) const;

  // Operations
  SimpleVector &operator=(const SimpleVector &v);

  void add(Number a, const SimpleVector &x);

  SimpleVector &operator*=(const Number factor);

  Number operator*(const SimpleVector &x) const;

  Number l2_norm() const;

  SimpleVector &operator=(const Number s);

  void print() const;

private:
  size_t _dim;
  std::vector<Number> _data;
};

} // namespace lanczos

} // namespace mfmg

#endif
