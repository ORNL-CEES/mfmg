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

#ifndef MFMG_LANCZOS_SIMPLEOP_HPP
#define MFMG_LANCZOS_SIMPLEOP_HPP

#include <mfmg/common/exceptions.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace mfmg
{

//-----------------------------------------------------------------------------
/// \brief Simple test operator
///
///        A diagonal matrix with equally spaced eigenvalues of some
///        multiplicity.

template <typename VectorType_>
class SimpleOperator : public OperatorBase<VectorType_>
{
public:
  // Typedefs
  using VectorType = VectorType_;

  SimpleOperator(size_t dim, size_t multiplicity = 1);

  SimpleOperator(const SimpleOperator<VectorType> &) = delete;
  SimpleOperator<VectorType> &
  operator=(const SimpleOperator<VectorType> &) = delete;

  std::vector<double> get_evals() const;

  void vmult(VectorType &y, VectorType const &x) const override;

  std::vector<double> get_diag_elements() const;

  size_t m() const override { return _dim; }
  size_t n() const override { return _dim; }

private:
  size_t _dim;
  size_t _multiplicity;
};

} // namespace mfmg

#endif
