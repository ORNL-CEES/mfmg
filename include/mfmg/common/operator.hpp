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

#ifndef MFMG_OPERATOR_HPP
#define MFMG_OPERATOR_HPP

#include <memory>

namespace mfmg
{
enum class OperatorMode
{
  NO_TRANS,
  TRANS
};

template <typename VectorType>
class OperatorBase
{
public:
  using operator_type = OperatorBase<VectorType>;
  using vector_type = VectorType;

  virtual size_t m() const = 0;
  virtual size_t n() const = 0;

  virtual void vmult(VectorType &y, VectorType const &x) const = 0;
};

template <typename VectorType>
class Operator
{
public:
  using operator_type = Operator<VectorType>;
  using vector_type = VectorType;

  virtual ~Operator() = default;

  virtual void apply(vector_type const &x, vector_type &y,
                     OperatorMode mode = OperatorMode::NO_TRANS) const = 0;

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
