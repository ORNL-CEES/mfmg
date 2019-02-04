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
namespace lanczos
{

//-----------------------------------------------------------------------------
/// \brief Lanczos solver

template <typename OperatorType_>
class Lanczos
{

public:
  // Typedefs

  typedef OperatorType_ OperatorType;
  typedef typename OperatorType::VectorType VectorType;
  typedef typename OperatorType::ScalarType ScalarType;
  typedef typename std::vector<ScalarType> Scalars_t;
  typedef typename OperatorType::Vectors_t Vectors_t;

  // Ctor/dtor

  Lanczos(OperatorType const &op, boost::property_tree::ptree const &params);
  ~Lanczos();

  // Accessors

  ScalarType get_eval(int i) const;

  VectorType *get_evec(int i) const;
  Vectors_t get_evecs() const;

  int num_evecs() const { return _num_requested; }

  // Operations

  void solve();

  void details_solve_lanczos(const int num_requested,
                             VectorType const &initial_guess, Scalars_t &evals,
                             Vectors_t &evecs);

private:
  OperatorType const &_op;         // reference to operator object to use
  int _num_requested;              // number of eigenpairs to calculate
  int _maxit;                      // maximum number of lanc interations
  double _tol;                     // convergence tolerance for eigenvalue
  unsigned int _percent_overshoot; // allowed iteration count overshoot from
                                   // less frequent stopping tests
  unsigned int _verbosity;         // verbosity of output

  Scalars_t _evals; // (approximate) eigenvals
  Vectors_t _evecs; // (approximate) eigenvecs of full operator

  void details_calc_tridiag_epairs(Scalars_t const &t_maindiag,
                                   Scalars_t const &t_offdiag,
                                   const int num_requested, Scalars_t &evals,
                                   Scalars_t &evecs);

  bool details_check_convergence(ScalarType beta, const int num_requested,
                                 double tol, Scalars_t const &evecs);

  void details_calc_evecs(const int num_requested, const int n,
                          Vectors_t const &lanc_vectors,
                          Scalars_t const &evecs_tridiag, Vectors_t &evecs);

  // Disallowed methods

  Lanczos(const Lanczos<OperatorType> &);
  void operator=(const Lanczos<OperatorType> &);
};

} // namespace lanczos

} // namespace mfmg

#endif
