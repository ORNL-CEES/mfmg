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

#ifndef MFMG_LANCZOS_LANCZOS_HPP
#define MFMG_LANCZOS_LANCZOS_HPP

#include <boost/property_tree/ptree.hpp>

#include <memory>
#include <vector>

namespace mfmg
{

//-----------------------------------------------------------------------------
/// \brief Lanczos solver

template <typename OperatorType, typename VectorType>
class Lanczos
{
public:
  Lanczos(OperatorType const &op);

  Lanczos(Lanczos<OperatorType, VectorType> const &) = delete;
  Lanczos<OperatorType, VectorType> &
  operator=(Lanczos<OperatorType, VectorType> const &) = delete;

  // Operations
  std::tuple<std::vector<double>, std::vector<VectorType>>
  solve(boost::property_tree::ptree const &params,
        VectorType initial_guess) const;

private:
  OperatorType const &_op; // reference to operator object to use

  template <typename FullOperatorType>
  static std::tuple<std::vector<double>, std::vector<VectorType>>
  details_solve_lanczos(FullOperatorType const &op, const int num_requested,
                        boost::property_tree::ptree const &params,
                        VectorType const &initial_guess);

  static std::tuple<std::vector<double>, std::vector<double>>
  details_calc_tridiag_epairs(std::vector<double> const &main_diagonal,
                              std::vector<double> const &sub_diagonal,
                              const int num_requested);

  static bool details_check_convergence(double beta, const int num_evecs,
                                        const int num_requested, double tol,
                                        std::vector<double> const &evecs);

  static std::vector<VectorType>
  details_calc_evecs(const int num_requested, const int n,
                     std::vector<VectorType> const &lanc_vectors,
                     std::vector<double> const &evecs_tridiag);

  static void details_set_initial_guess(VectorType &v, int seed = 0);
};

} // namespace mfmg

#endif
