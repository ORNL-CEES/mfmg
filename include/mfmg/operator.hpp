/*************************************************************************
 * Copyright (c) 2017-2018 by the mfmg authors                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#ifndef MFMG_OPERATOR_HPP
#define MFMG_OPERATOR_HPP

#include <memory>

namespace mfmg
{
template <typename VectorType>
class Operator
{
public:
  using operator_type = Operator<VectorType>;
  using vector_type = VectorType;

  virtual void apply(vector_type const &x, vector_type &y) const = 0;

  virtual std::shared_ptr<operator_type> transpose() const = 0;

  virtual std::shared_ptr<operator_type>
  multiply(std::shared_ptr<operator_type const> b) const = 0;

  virtual std::shared_ptr<operator_type>
  multiply_transpose(std::shared_ptr<operator_type const> b) const = 0;

  virtual std::shared_ptr<vector_type> build_domain_vector() const = 0;

  virtual std::shared_ptr<vector_type> build_range_vector() const = 0;

  virtual size_t grid_complexity() const = 0;

  virtual size_t operator_complexity() const = 0;
};
} // namespace mfmg

#endif
