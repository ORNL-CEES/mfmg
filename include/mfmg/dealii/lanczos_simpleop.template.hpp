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

#ifndef MFMG_LANCZOS_SIMPLEOP_TEMPLATE_HPP
#define MFMG_LANCZOS_SIMPLEOP_TEMPLATE_HPP

#include <cassert>

#include "lanczos_simpleop.hpp"

namespace mfmg
{
namespace lanczos
{

//-----------------------------------------------------------------------------
/// \brief Simple test operator: constructor

template<typename Vector_t>
SimpleOp<Vector_t>::SimpleOp(size_t dim, size_t multiplicity)
  : dim_(dim)
  , multiplicity_(multiplicity) {
  //assert(this->dim_ >= 0);
  assert(multiplicity > 0);
}

//-----------------------------------------------------------------------------
/// \brief Simple test operator: destructor

template<typename Vector_t>
SimpleOp<Vector_t>::~SimpleOp() {
}

//-----------------------------------------------------------------------------
/// \brief Simple test operator: apply operator to a vector

template<typename Vector_t>
void SimpleOp<Vector_t>::apply(Vector_t& vout, const Vector_t& vin) const {

  for (int i=0; i<this->dim_; ++i) {
    vout.elt(i) = this->diag_value_(i) * vin.const_elt(i);
  }
}

} // namespace lanczos

} // namespace mfmg

#endif
