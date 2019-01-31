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

template <typename Scalar_t>
SimpleVector<Scalar_t>::SimpleVector(size_t dim) : dim_(dim)
{
  // assert(dim_ >= 0);

  // NOTE: we are implementing basic operations, e.g., BLAS-1, on an
  // STL standard vector here.  This can be replaced if desired, e.g.,
  // something that lives on a GPU.

  // ISSUE: performance and memory management behaviors are
  // dependent on the specifics of the std::vector implementation.

  data_.resize(dim_);
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: destructor

template <typename Scalar_t>
SimpleVector<Scalar_t>::~SimpleVector()
{

  data_.resize(0);
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: element accessor

template <typename Scalar_t>
Scalar_t &SimpleVector<Scalar_t>::elt(size_t i)
{
  // assert(i >= 0);
  assert(i < this->dim_);

  return data_.data()[i];
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: (const) element accessor

template <typename Scalar_t>
Scalar_t SimpleVector<Scalar_t>::const_elt(size_t i) const
{
  // assert(i >= 0);
  assert(i < this->dim_);

  return data_[i];
}

#if 0
//-----------------------------------------------------------------------------

template<typename Scalar_t>
SimpleVector<Scalar_t> SimpleVector<Scalar_t>::copy() const {

  SimpleVector<Scalar_t> v(this->dim());
  v.copy(*this);
  return v;
}
#endif

//-----------------------------------------------------------------------------
/// \brief Simple vector: copy contents of one vector into another

template <typename Scalar_t>
void SimpleVector<Scalar_t>::copy(const SimpleVector &x)
{

  for (int i = 0; i < this->dim_; ++i)
  {
    this->elt(i) = x.const_elt(i);
  }
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: axpy operation

template <typename Scalar_t>
void SimpleVector<Scalar_t>::axpy(Scalar_t a, const SimpleVector &x)
{

  for (int i = 0; i < this->dim_; ++i)
  {
    this->elt(i) += a * x.const_elt(i);
  }
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: scale operation

template <typename Scalar_t>
void SimpleVector<Scalar_t>::scal(Scalar_t a)
{

  for (int i = 0; i < this->dim_; ++i)
  {
    this->elt(i) *= a;
  }
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: dot product operation

template <typename Scalar_t>
Scalar_t SimpleVector<Scalar_t>::dot(const SimpleVector &x) const
{

  Scalar_t sum = (Scalar_t)0;

  for (int i = 0; i < this->dim_; ++i)
  {
    sum += this->const_elt(i) * x.const_elt(i);
  }

  return sum;
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: 2-norm product operation

template <typename Scalar_t>
Scalar_t SimpleVector<Scalar_t>::nrm2() const
{

  Scalar_t vdotv = this->dot(*this);

  return (Scalar_t)sqrt((double)vdotv);
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: set elements to zero

template <typename Scalar_t>
void SimpleVector<Scalar_t>::set_zero()
{

  // for (int i=0; i<this->dim_; ++i) {
  //  this->elt(i) = (Scalar_t)0;
  //}
  std::generate(data_.begin(), data_.end(), [&]() { return (Scalar_t)0; });
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: set elements to random numbers

template <typename Scalar_t>
void SimpleVector<Scalar_t>::set_random(int seed, double multiplier,
                                        double cmultiplier)
{

  // Compute a x + b y where x has uniformly distributed random entries
  // in [0,1] and y entries are all 1.

  // std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937 gen(seed);

  std::uniform_real_distribution<double> dis(0, 1);
  std::generate(data_.begin(), data_.end(), [&]() {
    return (Scalar_t)(multiplier * dis(gen) + cmultiplier);
  });
  // std::generate(this->data_.begin(), this->data_.end(), [&](){
  //  const int v = dis(gen);
  //  std::cout << v << "\n";
  //  return (Scalar_t)v;
  //});
}

//-----------------------------------------------------------------------------
/// \brief Simple vector: print to cout

template <typename Scalar_t>
void SimpleVector<Scalar_t>::print() const
{

  for (int i = 0; i < dim_; ++i)
  {
    std::cout << data_[i] << " ";
  }
  std::cout << std::endl;
}

} // namespace lanczos

} // namespace mfmg

#endif
