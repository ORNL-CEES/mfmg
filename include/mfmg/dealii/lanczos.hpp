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

template <typename VectorType>
class Lanczos
{
public:
  // Typedefs
  using vector_type = VectorType;
  using ScalarType = typename VectorType::value_type;
  using OperatorType = Operator<VectorType>;

  typedef typename std::vector<ScalarType> Scalars_t;
  typedef typename std::vector<VectorType *> Vectors_t;

  // Ctor/dtor

  Lanczos(OperatorType const &op, boost::property_tree::ptree const &params);
  ~Lanczos();

  // Accessors

  ScalarType get_eval(int i) const;

  VectorType *get_evec(int i) const;
  Vectors_t get_evecs() const;

  int num_evecs() const
  {
    return (_is_deflated ? _num_evecs_per_cycle * _num_cycles : _num_requested);
  }

  // Operations

  void solve();

private:
  OperatorType const &_op;         // reference to operator object to use
  bool _is_deflated;               // mode
  int _num_requested;              // number of eigenpairs to calculate
  int _num_evecs_per_cycle;        // number of eigs to calc per lanc solve
  int _num_cycles;                 // number of lanczos solves
  int _maxit;                      // maximum number of lanc interations
  double _tol;                     // convergence tolerance for eigenvalue
  unsigned int _percent_overshoot; // allowed iteration count overshoot from
                                   // less frequent stopping tests
  unsigned int _verbosity;         // verbosity of output

  Scalars_t _evals; // (approximate) eigenvals of full operator
  Vectors_t _evecs; // (approximate) eigenvecs of full operator

  void details_solve_lanczos(Operator<VectorType> const &op,
                             const int num_requested,
                             VectorType const &initial_guess, Scalars_t &evals,
                             Vectors_t &evecs);

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

  Lanczos(const Lanczos<VectorType> &);
  void operator=(const Lanczos<VectorType> &);
};

} // namespace mfmg

#endif
