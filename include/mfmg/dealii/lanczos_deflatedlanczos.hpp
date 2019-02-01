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

#ifndef MFMG_LANCZOS_DEFLATEDLANCZOS_HPP
#define MFMG_LANCZOS_DEFLATEDLANCZOS_HPP

#include <vector>

namespace mfmg
{
namespace lanczos
{

//-----------------------------------------------------------------------------
/// \brief Deflated Lanczos solver
///
///        The Lanczos solver is called multiple times.  After each Lanczos
///        solve, the operator is modified to deflate out all previously
///        computed (approximate) eigenvectors.  This is meant to deal with
///        possible eigenvalue multiplicities.

template <typename OperatorType_>
class DeflatedLanczos
{

public:
  // Typedefs

  typedef OperatorType_ OperatorType;
  typedef typename OperatorType::VectorType VectorType;
  typedef typename OperatorType::ScalarType ScalarType;
  typedef typename std::vector<ScalarType> Scalars_t;
  typedef typename std::vector<VectorType *> Vectors_t;

  // Ctor/dtor

  DeflatedLanczos(OperatorType const &op,
                  boost::property_tree::ptree const &params);
  ~DeflatedLanczos();

  // Accessors

  ScalarType get_eval(int i) const;

  VectorType *get_evec(int i) const;

  int num_evecs() const { return _num_evecs_per_cycle * _num_cycles; }

  // Operations

  void solve();

private:
  const OperatorType &_op;
  int _num_evecs_per_cycle; // number of eigs to calc per lanc solve
  int _num_cycles;          // number of lanczos solves
  int _maxit;               // maximum number of lanc interations
  double _tol;              // convergence tolerance for eigenvalue
  unsigned int _percent_overshoot;
  // allowed iteration count overshoot from
  // less frequent stopping tests
  unsigned int _verbosity; // verbosity of output

  size_t _dim; // operator and vector dimension

  std::vector<ScalarType> _evals; // (approximate) eigenvals of full operator
  Vectors_t _evecs;               // (approximate) eigenvecs of full operator

  // Disallowed methods

  DeflatedLanczos(const DeflatedLanczos<OperatorType> &);
  void operator=(const DeflatedLanczos<OperatorType> &);
};

} // namespace lanczos

} // namespace mfmg

#endif
