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

template<typename Op_t_>
class DeflatedLanczos {

  public:

    // Typedefs

    typedef Op_t_ Op_t;
    typedef typename Op_t::Vector_t Vector_t;
    typedef typename Op_t::Scalar_t Scalar_t;
    typedef typename std::vector<Scalar_t> Scalars_t;
    typedef typename std::vector<Vector_t*> Vectors_t;

    // Ctor/dtor

    DeflatedLanczos(Op_t& op, int num_evecs_per_cycle, int num_cycles,
            int maxit, double tol, unsigned int percent_overshoot = 0,
            unsigned int verbosity = 0);
    ~DeflatedLanczos();

    // Accessors

    Scalar_t get_eval(int i) const;

    Vector_t* get_evec(int i) const;

    int num_evecs() const {return num_evecs_per_cycle_ * num_cycles_;}

    // Operations

    void solve();

  private:

    const Op_t& op_;
    const int num_evecs_per_cycle_; // number of eigs to calc per lanc solve
    const int num_cycles_;          // number of lanczos solves
    const int maxit_;               // maximum number of lanc interations
    const double tol_;              // convergence tolerance for eigenvalue
    const unsigned int percent_overshoot_;
                                    // allowed iteration count overshoot from
                                    // less frequent stopping tests
    const unsigned int verbosity_;  // verbosity of output

    size_t dim_;                    // operator and vector dimension

    std::vector<Scalar_t> evals_;   // (approximate) eigenvals of full operator
    Vectors_t evecs_;               // (approximate) eigenvecs of full operator

    // Disallowed methods

    DeflatedLanczos(const DeflatedLanczos<Op_t>&);
    void operator=(const  DeflatedLanczos<Op_t>&);
};

} // namespace lanczos

} // namespace mfmg

#endif
