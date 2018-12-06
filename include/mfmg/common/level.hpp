/**************************************************************************
 * Copyright (c) 2017-2018 by the mfmg authors                            *
 * All rights reserved.                                                   *
 *                                                                        *
 * This file is part of the mfmg libary. mfmg is distributed under op BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the  *
 * top-level directory                                                    *
 *                                                                        *
 * SPDX-License-Identifier: BSD-3-Clause                                  *
 **************************************************************************/

#ifndef MFMG_LEVEL_HPP
#define MFMG_LEVEL_HPP

#include <mfmg/common/operator.hpp>
#include <mfmg/common/smoother.hpp>
#include <mfmg/common/solver.hpp>

namespace mfmg
{

template <typename VectorType>
class Level
{
public:
  using vector_type = VectorType;
  using operator_type = Operator<VectorType>;

  std::shared_ptr<operator_type const> get_operator() const
  {
    return _operator;
  }

  std::shared_ptr<operator_type const> get_restrictor() const
  {
    return _restrictor;
  }

  std::shared_ptr<Smoother<vector_type> const> get_smoother() const
  {
    return _smoother;
  }

  std::shared_ptr<Solver<vector_type> const> get_solver() const
  {
    return _solver;
  }

  void set_operator(std::shared_ptr<operator_type const> op) { _operator = op; }

  void set_restrictor(std::shared_ptr<operator_type const> r)
  {
    _restrictor = r;
  }

  void set_smoother(std::shared_ptr<Smoother<vector_type> const> s)
  {
    _smoother = s;
  }

  void set_solver(std::shared_ptr<Solver<vector_type> const> s) { _solver = s; }

  std::shared_ptr<vector_type> build_vector() const
  {
    auto a = get_operator();
    ASSERT(
        a != nullptr,
        "build_vector() can only be called after the level operator was set.");
    return a->build_domain_vector();
  }

private:
  std::shared_ptr<operator_type const> _operator, _restrictor;
  std::shared_ptr<Smoother<vector_type> const> _smoother;
  std::shared_ptr<Solver<vector_type> const> _solver;
};
} // namespace mfmg

#endif
