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

//-----------------------------------------------------------------------------
/// \brief Simple test operator: constructor

template <typename VectorType>
SimpleOperator<VectorType>::SimpleOperator(size_t dim, size_t multiplicity)
    : _dim(dim), _multiplicity(multiplicity)
{
  // assert(this->_dim >= 0);
  assert(multiplicity > 0);
}

//-----------------------------------------------------------------------------
/// \brief Simple test operator: destructor

template <typename VectorType>
SimpleOperator<VectorType>::~SimpleOperator()
{
}

//-----------------------------------------------------------------------------
/// \brief Simple test operator: apply operator to a vector

template <typename VectorType>
void SimpleOperator<VectorType>::apply(VectorType const &x, VectorType &y,
                                       OperatorMode mode) const
{

  for (int i = 0; i < this->_dim; ++i)
  {
    y[i] = this->diag_value_(i) * x[i];
  }
}

} // namespace mfmg

#endif
