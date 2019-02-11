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

#include <mfmg/common/operator.hpp>

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
  // Typedefs
  using ScalarType = typename VectorType::value_type;

  // Ctor/dtor
  Lanczos(OperatorType const &op, boost::property_tree::ptree const &params);
  ~Lanczos() {}

  // Accessors
  std::vector<ScalarType> const &get_evals() const;
  ScalarType get_eval(int i) const;

  VectorType const &get_evec(int i) const;
  std::vector<VectorType> const &get_evecs() const;

  int num_evecs() const
  {
    return (_is_deflated ? _num_evecs_per_cycle * _num_cycles : _num_requested);
  }

  // Operations

  void solve();

private:
  OperatorType const &_op;  // reference to operator object to use
  bool _is_deflated;        // mode
  int _num_requested;       // number of eigenpairs to calculate
  int _num_evecs_per_cycle; // number of eigs to calc per lanc solve
  int _num_cycles;          // number of lanczos solves
  int _maxit;               // maximum number of lanc interations
  double _tol;              // convergence tolerance for eigenvalue
  int _percent_overshoot;   // allowed iteration count overshoot from
                            // less frequent stopping tests
  unsigned int _verbosity;  // verbosity of output

  std::vector<ScalarType> _evals; // (approximate) eigenvals of full operator
  std::vector<VectorType> _evecs; // (approximate) eigenvecs of full operator

  template <typename FullOperatorType>
  void details_solve_lanczos(FullOperatorType const &op,
                             const int num_requested,
                             VectorType const &initial_guess,
                             std::vector<ScalarType> &evals,
                             std::vector<VectorType> &evecs);

  void details_calc_tridiag_epairs(std::vector<ScalarType> const &t_maindiag,
                                   std::vector<ScalarType> const &t_offdiag,
                                   const int num_requested,
                                   std::vector<ScalarType> &evals,
                                   std::vector<ScalarType> &evecs);

  bool details_check_convergence(ScalarType beta, const int num_requested,
                                 double tol,
                                 std::vector<ScalarType> const &evecs);

  void details_calc_evecs(const int num_requested, const int n,
                          std::vector<VectorType> const &lanc_vectors,
                          std::vector<ScalarType> const &evecs_tridiag,
                          std::vector<VectorType> &evecs);

  // Disallowed methods

  Lanczos(const Lanczos<OperatorType, VectorType> &);
  void operator=(const Lanczos<OperatorType, VectorType> &);
};

} // namespace mfmg

#endif
