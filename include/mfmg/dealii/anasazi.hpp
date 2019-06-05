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

#ifndef MFMG_ANASAZI_HPP
#define MFMG_ANASAZI_HPP

#include <boost/property_tree/ptree.hpp>

#include <vector>

namespace mfmg
{

template <typename OperatorType, typename VectorType>
class AnasaziSolver
{
public:
  // Constructors
  AnasaziSolver(OperatorType const &op);

  AnasaziSolver(AnasaziSolver<OperatorType, VectorType> const &) = delete;
  AnasaziSolver<OperatorType, VectorType> &
  operator=(AnasaziSolver<OperatorType, VectorType> const &) = delete;

  // Solve the eigenproblem
  std::tuple<std::vector<double>, std::vector<VectorType>>
  solve(boost::property_tree::ptree const &params,
        VectorType initial_guess) const;

private:
  OperatorType const &_op; // reference to operator object to use
};

} // namespace mfmg

#endif
