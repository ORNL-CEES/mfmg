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

template <typename ScalarType>
SimpleVector<ScalarType>::SimpleVector(size_t dim) : _dim(dim)
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

template <typename ScalarType>
SimpleVector<ScalarType>::~SimpleVector()
{

  _data.resize(0);
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: element accessor

template <typename ScalarType>
ScalarType &SimpleVector<ScalarType>::elt(size_t i)
{
  // assert(i >= 0);
  assert(i < this->_dim);

  return _data.data()[i];
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: (const) element accessor

template <typename ScalarType>
ScalarType SimpleVector<ScalarType>::const_elt(size_t i) const
{
  // assert(i >= 0);
  assert(i < this->_dim);

  return _data[i];
}

#if 0
//-----------------------------------------------------------------------------

template<typename ScalarType>
SimpleVector<ScalarType> SimpleVector<ScalarType>::copy() const {

  SimpleVector<ScalarType> v(this->dim());
  v.copy(*this);
  return v;
}
#endif

//-----------------------------------------------------------------------------
/// \brief Simple vector: copy contents of one vector into another

template <typename ScalarType>
void SimpleVector<ScalarType>::copy(const SimpleVector &x)
{

  for (int i = 0; i < this->_dim; ++i)
  {
    this->elt(i) = x.const_elt(i);
  }
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: axpy operation

template <typename ScalarType>
void SimpleVector<ScalarType>::axpy(ScalarType a, const SimpleVector &x)
{

  for (int i = 0; i < this->_dim; ++i)
  {
    this->elt(i) += a * x.const_elt(i);
  }
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: scale operation

template <typename ScalarType>
void SimpleVector<ScalarType>::scal(ScalarType a)
{

  for (int i = 0; i < this->_dim; ++i)
  {
    this->elt(i) *= a;
  }
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: dot product operation

template <typename ScalarType>
ScalarType SimpleVector<ScalarType>::dot(const SimpleVector &x) const
{

  ScalarType sum = (ScalarType)0;

  for (int i = 0; i < this->_dim; ++i)
  {
    sum += this->const_elt(i) * x.const_elt(i);
  }

  return sum;
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: 2-norm product operation

template <typename ScalarType>
ScalarType SimpleVector<ScalarType>::nrm2() const
{

  ScalarType vdotv = this->dot(*this);

  return (ScalarType)sqrt((double)vdotv);
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: set elements to zero

template <typename ScalarType>
void SimpleVector<ScalarType>::set_zero()
{

  // for (int i=0; i<this->_dim; ++i) {
  //  this->elt(i) = (ScalarType)0;
  //}
  std::generate(_data.begin(), _data.end(), [&]() { return (ScalarType)0; });
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: set elements to random numbers

template <typename ScalarType>
void SimpleVector<ScalarType>::set_random(int seed, double multiplier,
                                          double cmultiplier)
{

  // Compute a x + b y where x has uniformly distributed random entries
  // in [0,1] and y entries are all 1.

  // std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937 gen(seed);

  std::uniform_real_distribution<double> dis(0, 1);
  std::generate(_data.begin(), _data.end(), [&]() {
    return (ScalarType)(multiplier * dis(gen) + cmultiplier);
  });
  // std::generate(this->_data.begin(), this->_data.end(), [&](){
  //  const int v = dis(gen);
  //  std::cout << v << "\n";
  //  return (ScalarType)v;
  //});
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: print to cout

template <typename ScalarType>
void SimpleVector<ScalarType>::print() const
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
