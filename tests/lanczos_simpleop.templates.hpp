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

#ifndef MFMG_LANCZOS_SIMPLEOP_TEMPLATE_HPP
#define MFMG_LANCZOS_SIMPLEOP_TEMPLATE_HPP

#include <cassert>

#include "lanczos_simpleop.hpp"

namespace mfmg
{

/// \brief Simple test operator: constructor
template <typename VectorType>
SimpleOperator<VectorType>::SimpleOperator(size_t dim, size_t multiplicity)
    : _dim(dim), _multiplicity(multiplicity)
{
  assert(_dim >= 0);
  assert(multiplicity > 0);
}

template <typename VectorType>
std::vector<double> SimpleOperator<VectorType>::get_evals() const
{
  std::vector<double> evals(_dim);
  for (size_t i = 0; i < _dim; i++)
    evals[i] = diag_value_(i);

  return evals;
}

/// \brief Simple test operator: apply operator to a vector
template <typename VectorType>
void SimpleOperator<VectorType>::vmult(VectorType &y, VectorType const &x) const
{
  for (size_t i = 0; i < _dim; ++i)
    y[i] = diag_value_(i) * x[i];
}

} // namespace mfmg

#endif
