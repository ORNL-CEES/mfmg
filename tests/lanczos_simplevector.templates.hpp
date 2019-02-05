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

#ifndef MFMG_LANCZOS_SIMPLEVECTOR_TEMPLATE_HPP
#define MFMG_LANCZOS_SIMPLEVECTOR_TEMPLATE_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#include "lanczos_simplevector.hpp"

namespace mfmg
{
namespace lanczos
{

//-----------------------------------------------------------------------------
/// \brief Simple vector: constructor

template <typename Number>
SimpleVector<Number>::SimpleVector(const size_t n) : _dim(n)
{
  // assert(_dim >= 0);

  // NOTE: we are implementing basic operations, e.g., BLAS-1, on an
  // STL standard vector here.  This can be replaced if desired, e.g.,
  // something that lives on a GPU.

  // ISSUE: performance and memory management behaviors are
  // dependent on the specifics of the std::vector implementation.

  _data.resize(_dim);
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: destructor

template <typename Number>
SimpleVector<Number>::~SimpleVector()
{

  _data.resize(0);
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: element accessor

template <typename Number>
Number &SimpleVector<Number>::operator[](size_t i)
{
  assert(i < this->_dim);

  return _data.data()[i];
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: (const) element accessor

template <typename Number>
Number SimpleVector<Number>::operator[](size_t i) const
{
  assert(i < this->_dim);

  return _data[i];
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: copy contents of one vector into another

template <typename Number>
SimpleVector<Number> &SimpleVector<Number>::
operator=(const SimpleVector<Number> &x)
{
  for (int i = 0; i < this->_dim; ++i)
  {
    (*this)[i] = x[i];
  }
  return *this;
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: axpy operation

template <typename Number>
void SimpleVector<Number>::add(Number a, const SimpleVector &x)
{

  for (int i = 0; i < this->_dim; ++i)
  {
    (*this)[i] += a * x[i];
  }
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: scale operation

template <typename Number>
SimpleVector<Number> &SimpleVector<Number>::operator*=(const Number factor)
{

  for (int i = 0; i < this->_dim; ++i)
  {
    (*this)[i] *= factor;
  }
  return *this;
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: dot product operation

template <typename Number>
Number SimpleVector<Number>::operator*(const SimpleVector<Number> &x) const
{

  Number sum = (Number)0;

  for (int i = 0; i < this->_dim; ++i)
  {
    sum += (*this)[i] * x[i];
  }

  return sum;
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: 2-norm product operation

template <typename Number>
Number SimpleVector<Number>::l2_norm() const
{

  Number vdotv = (*this) * (*this);

  return (Number)sqrt((double)vdotv);
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: set elements to zero

template <typename Number>
SimpleVector<Number> &SimpleVector<Number>::operator=(const Number s)
{
  std::generate(_data.begin(), _data.end(), [&]() { return s; });
  return *this;
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: print to cout

template <typename Number>
void SimpleVector<Number>::print() const
{

  for (int i = 0; i < _dim; ++i)
  {
    std::cout << _data[i] << " ";
  }
  std::cout << std::endl;
}

} // namespace lanczos

} // namespace mfmg

#endif
