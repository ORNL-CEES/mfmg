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

#ifndef MFMG_SMOOTHER_HPP
#define MFMG_SMOOTHER_HPP

#include <mfmg/operator.hpp>

#include <boost/property_tree/ptree.hpp>

#include <memory>

namespace mfmg
{
template <typename VectorType>
class Smoother
{
public:
  using vector_type = VectorType;

  Smoother(std::shared_ptr<Operator<vector_type> const> op,
           std::shared_ptr<boost::property_tree::ptree const> params)
      : _operator(op), _params(params)
  {
  }

  virtual void apply(vector_type const &x, vector_type &y) const = 0;

  virtual ~Smoother() = default;

protected:
  std::shared_ptr<Operator<vector_type> const> _operator;
  std::shared_ptr<boost::property_tree::ptree const> _params;
};
} // namespace mfmg

#endif
