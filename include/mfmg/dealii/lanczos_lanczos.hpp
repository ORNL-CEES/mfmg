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
  void solve(const VectorType &guess);

private:
  OperatorType const &_op;         // reference to operator object to use
  int _num_requested;              // number of eigenpairs to calculate
  int _maxit;                      // maximum number of lanc interations
  double _tol;                     // convergence tolerance for eigenvalue
  unsigned int _percent_overshoot; // allowed iteration count overshoot from
                                   // less frequent stopping tests
  unsigned int _verbosity;         // verbosity of output

  size_t _dim;         // operator and vector dimension
  size_t _dim_tridiag; // dimension of tridiag matrix

  Vectors_t _lanc_vectors; // lanczos vectors

  std::vector<ScalarType> _evals; // (approximate) eigenvals
  std::vector<ScalarType> _evecs_tridiag;
  // eigenvecs of tridiag matrix,
  // stored in flat array

  Vectors_t _evecs; // (approximate) eigenvecs of full operator

  void calc_tridiag_epairs_(int it, Scalars_t &t_maindiag,
                            Scalars_t &t_offdiag);

  bool check_convergence_(ScalarType beta);

  void calc_evecs_();

  // Disallowed methods

  Lanczos(const Lanczos<OperatorType> &);
  void operator=(const Lanczos<OperatorType> &);
};

} // namespace lanczos

} // namespace mfmg

#endif
