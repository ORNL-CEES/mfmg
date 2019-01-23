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

#ifndef _LANCZOS_LANCZOS_HPP_
#define _LANCZOS_LANCZOS_HPP_

#include <vector>

namespace mfmg::lanczos
{

//-----------------------------------------------------------------------------

template<typename Op_t_>
class Lanczos {

  public:

    // Typedefs

    typedef Op_t_ Op_t;
    typedef typename Op_t::Vector_t Vector_t;
    typedef typename Op_t::Scalar_t Scalar_t;
    typedef typename std::vector<Scalar_t> Scalars_t;
    typedef typename Op_t::Vectors_t Vectors_t;

    // Ctor/dtor

    Lanczos(const Op_t& op, int num_requested, int maxit, double tol,
            unsigned int percent_overshoot = 0, unsigned int verbosity = 0);
    ~Lanczos();

    // Accessors

    Scalar_t get_eval(int i) const;

    Vector_t* get_evec(int i) const;
    Vectors_t get_evecs() const;

    int num_evecs() const {return num_requested_;}

    // Operations

    void solve();
    void solve(const Vector_t& guess);

  private:

    const Op_t& op_;
    const int num_requested_;
    const int maxit_;
    const double tol_;
    const unsigned int percent_overshoot_;
    const unsigned int verbosity_;

    size_t dim_;
    size_t dim_tridiag_;

    Vectors_t lanc_vectors_;

    std::vector<Scalar_t> evals_;
    std::vector<Scalar_t> evecs_tridiag_;

    Vectors_t evecs_;

    void calc_tridiag_epairs_(int it, Scalars_t& t_maindiag,
                              Scalars_t& t_offdiag);

    bool check_convergence_(Scalar_t beta);

    void calc_evecs_();

    // Disallowed methods

    Lanczos(       const Lanczos<Op_t>&);
    void operator=(const Lanczos<Op_t>&);
};

//-----------------------------------------------------------------------------

} // namespace mfmg::lanczos

#endif // _LANCZOS_LANCZOS_HPP_

//=============================================================================
