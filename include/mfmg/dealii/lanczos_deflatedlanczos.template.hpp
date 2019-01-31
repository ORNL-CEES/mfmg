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

#include <cassert>
#include <algorithm>
#include <vector>
#include <iostream>

#include "cblas.h"
#include "lapacke.h"

#include "lanczos_deflatedop.template.hpp"
#include "lanczos_lanczos.template.hpp"
#include "lanczos_deflatedlanczos.hpp"

namespace mfmg::lanczos
{

//-----------------------------------------------------------------------------
/// \brief Deflated Lanczos solver: constructor

template<typename Op_t>
DeflatedLanczos<Op_t>::DeflatedLanczos(Op_t& op, int num_evecs_per_cycle,
 int num_cycles, int maxit, double tol, unsigned int percent_overshoot,
 unsigned int verbosity)
  : op_(op)
  , num_evecs_per_cycle_(num_evecs_per_cycle)
  , num_cycles_(num_cycles)
  , maxit_(maxit)
  , tol_(tol)
  , percent_overshoot_(percent_overshoot)
  , verbosity_(verbosity)
  , dim_(op.dim()) {
  assert(this->num_evecs_per_cycle_ >= 1);
  assert(this->num_cycles_ >= 1);
  assert(this->maxit_ >= 0);
  assert(this->maxit_ >= this->num_evecs_per_cycle_ &&
         "maxit too small to produce required number of eigenvectors.");
  assert(this->tol_ >= 0.);
}

//-----------------------------------------------------------------------------
/// \brief Deflated Lanczos solver: destructor

template<typename Op_t>
DeflatedLanczos<Op_t>::~DeflatedLanczos() {
  for (int i = 0; i < evecs_.size(); ++i) {
    delete evecs_[i];
  }
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: accessor for (approximate) eigenvalue

template<typename Op_t>
typename Op_t::Scalar_t DeflatedLanczos<Op_t>::get_eval(int i) const {
  assert(i >= 0);
  assert(i < evals_.size());

  return evals_[i];
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: accessor for (approximate) eigenvector

template<typename Op_t>
typename Op_t::Vector_t* DeflatedLanczos<Op_t>::get_evec(int i) const {
  assert(i >= 0);
  assert(i < evecs_.size());

  // ISSUE: giving users pointer to internal data that shouldn't be modified.
  return evecs_[i];
}

//-----------------------------------------------------------------------------
/// \brief Lanczos solver: perform deflated Lanczos solve

template<typename Op_t>
void DeflatedLanczos<Op_t>::solve() {

  typedef DeflatedOp<Op_t> DOp;

  // Form deflated operator from original operator.

  DOp deflated_op(this->op_);

  // Loop over Lanczos solves.

  for (int cycle=0; cycle<num_cycles_; ++cycle) {

    if (verbosity_ > 0) {
      std::cout << "----------------------------------------"
                   "---------------------------------------" << std::endl;
      std::cout << "Lanczos solve " << cycle + 1 << ":" << std::endl;
    }

    // Perform Lanczos solve, initial guess is a lin comb of a
    // constant vector (to try to capture "smooth" eigenmodes
    // of PDE problems and a random vector based on different random
    // seeds.
    // ISSUE; should a different initial guess strategy be used.

    Lanczos<DOp> solver(deflated_op, num_evecs_per_cycle_, maxit_, tol_,
                        percent_overshoot_, verbosity_);

    typename Op_t::Vector_t guess(dim_);
    guess.set_random(cycle, 1., 1.);
    // Deflate initial guess.
    deflated_op.deflate(guess);

    solver.solve(guess);

    // Save the eigenpairs just calculated.

    // NOTE: throughout we use the term eigenpair (= eigenvector, eigenvalue),
    // though the precise terminology should be "approximate eigenpairs"
    // or "Ritz pairs."

    for (int i=0; i<num_evecs_per_cycle_; ++i) {
      evals_.push_back(solver.get_eval(i));
      evecs_.push_back(new Vector_t(dim_));
      evecs_[i]->copy(solver.get_evec(i));
    }

    // Add eigenvectors to the set of vectors being deflated out.

    if (cycle != num_cycles_ - 1) {
      deflated_op.add_deflation_vecs(solver.get_evecs());
    }

  } // cycle
}

//-----------------------------------------------------------------------------

} // namespace mfmg::lanczos

#endif // _LANCZOS_DEFLATEDLANCZOS_TEMPLATE_HPP_

//=============================================================================
