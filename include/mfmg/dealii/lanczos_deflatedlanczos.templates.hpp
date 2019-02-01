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

#ifndef MFMG_LANCZOS_DEFLATEDLANCZOS_TEMPLATE_HPP
#define MFMG_LANCZOS_DEFLATEDLANCZOS_TEMPLATE_HPP

#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "cblas.h"
#include "lanczos_deflatedlanczos.hpp"
#include "lanczos_deflatedop.templates.hpp"
#include "lanczos_lanczos.templates.hpp"

// This complex code has to be included before lapacke for the code to compile.
// Otherwise, it conflicts with boost or Kokkos.
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include "lapacke.h"

namespace mfmg
{
namespace lanczos
{

//-----------------------------------------------------------------------------
/// \brief Deflated Lanczos solver: constructor

template <typename OperatorType>
DeflatedLanczos<OperatorType>::DeflatedLanczos(
    OperatorType const &op, boost::property_tree::ptree const &params)
    : _op(op)
{
  _num_evecs_per_cycle = params.get<int>("num_eigenpairs_per_cycle");
  _num_cycles = params.get<int>("num_cycles");
  _maxit = params.get<int>("max_iterations");
  _tol = params.get<double>("tolerance");
  _percent_overshoot = params.get<int>("percent_overshoot", 0);
  _verbosity = params.get<unsigned int>("verbosity", 0);
  _dim = op.dim();

  assert(this->_num_evecs_per_cycle >= 1);
  assert(this->_num_cycles >= 1);
  assert(this->_maxit >= 0);
  assert(this->_maxit >= this->_num_evecs_per_cycle &&
         "maxit too small to produce required number of eigenvectors.");
  assert(this->_tol >= 0.);
}

//-----------------------------------------------------------------------------
/// \brief Deflated Lanczos solver: destructor

template <typename OperatorType>
DeflatedLanczos<OperatorType>::~DeflatedLanczos()
{
  for (int i = 0; i < _evecs.size(); ++i)
  {
    delete _evecs[i];
  }
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: accessor for (approximate) eigenvalue

template <typename OperatorType>
typename OperatorType::ScalarType
DeflatedLanczos<OperatorType>::get_eval(int i) const
{
  assert(i >= 0);
  assert(i < _evals.size());

  return _evals[i];
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: accessor for (approximate) eigenvector

template <typename OperatorType>
typename OperatorType::VectorType *
DeflatedLanczos<OperatorType>::get_evec(int i) const
{
  assert(i >= 0);
  assert(i < _evecs.size());

  // ISSUE: giving users pointer to internal data that shouldn't be modified.
  return _evecs[i];
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: perform deflated Lanczos solve

template <typename OperatorType>
void DeflatedLanczos<OperatorType>::solve()
{

  typedef DeflatedOp<OperatorType> DOp;

  // Form deflated operator from original operator.

  DOp deflated_op(this->_op);

  // Loop over Lanczos solves.

  for (int cycle = 0; cycle < _num_cycles; ++cycle)
  {

    if (_verbosity > 0)
    {
      std::cout << "----------------------------------------"
                   "---------------------------------------"
                << std::endl;
      std::cout << "Lanczos solve " << cycle + 1 << ":" << std::endl;
    }

    // Perform Lanczos solve, initial guess is a lin comb of a
    // constant vector (to try to capture "smooth" eigenmodes
    // of PDE problems and a random vector based on different random
    // seeds.
    // ISSUE; should a different initial guess strategy be used.

    boost::property_tree::ptree lanczos_params;
    lanczos_params.put("num_eigenpairs", _num_evecs_per_cycle);
    lanczos_params.put("max_iterations", _maxit);
    lanczos_params.put("tolerance", _tol);
    lanczos_params.put("percent_overshoot", _percent_overshoot);
    lanczos_params.put("verbosity", _verbosity);
    Lanczos<DOp> solver(deflated_op, lanczos_params);

    typename OperatorType::VectorType guess(_dim);
    guess.set_random(cycle, 1., 1.);
    // Deflate initial guess.
    deflated_op.deflate(guess);

    solver.solve(guess);

    // Save the eigenpairs just calculated.

    // NOTE: throughout we use the term eigenpair (= eigenvector, eigenvalue),
    // though the precise terminology should be "approximate eigenpairs"
    // or "Ritz pairs."

    for (int i = 0; i < _num_evecs_per_cycle; ++i)
    {
      _evals.push_back(solver.get_eval(i));
      _evecs.push_back(new VectorType(_dim));
      _evecs[i]->copy(solver.get_evec(i));
    }

    // Add eigenvectors to the set of vectors being deflated out.

    if (cycle != _num_cycles - 1)
    {
      deflated_op.add_deflation_vecs(solver.get_evecs());
    }

  } // cycle
}

} // namespace lanczos

} // namespace mfmg

#endif
