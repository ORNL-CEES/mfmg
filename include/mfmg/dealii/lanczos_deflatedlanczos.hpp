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

#ifndef _LANCZOS_DEFLATEDLANCZOS_HPP_
#define _LANCZOS_DEFLATEDLANCZOS_HPP_

#include <vector>

namespace mfmg::lanczos
{

//-----------------------------------------------------------------------------

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
    const int num_evecs_per_cycle_;
    const int num_cycles_;
    const int maxit_;
    const double tol_;
    const unsigned int percent_overshoot_;
    const unsigned int verbosity_;

    size_t dim_;

    std::vector<Scalar_t> evals_;
    Vectors_t evecs_;

    // Disallowed methods

    DeflatedLanczos(const DeflatedLanczos<Op_t>&);
    void operator=(const  DeflatedLanczos<Op_t>&);
};

//-----------------------------------------------------------------------------

} // namespace mfmg::lanczos

#endif // _LANCZOS_DEFLATEDLANCZOS_HPP_

//=============================================================================
