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

#ifndef MFMG_DEALII_SMOOTHER_HPP
#define MFMG_DEALII_SMOOTHER_HPP

#include <mfmg/common/operator.hpp>
#include <mfmg/common/smoother.hpp>

#include <deal.II/lac/trilinos_precondition.h>

#include <boost/property_tree/ptree.hpp>

#include <memory>

namespace mfmg
{
template <typename VectorType>
class DealIISmoother : public Smoother<VectorType>
{
public:
  using vector_type = VectorType;

  DealIISmoother(std::shared_ptr<Operator<vector_type> const> op,
                 std::shared_ptr<boost::property_tree::ptree const> params);

  void apply(vector_type const &b, vector_type &x) const override final;

private:
  std::unique_ptr<dealii::TrilinosWrappers::PreconditionBase> _smoother;
};
} // namespace mfmg

#endif
