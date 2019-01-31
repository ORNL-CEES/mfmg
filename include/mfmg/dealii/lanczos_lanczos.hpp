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

#include <vector>

namespace mfmg
{
namespace lanczos
{

//-----------------------------------------------------------------------------
/// \brief Lanczos solver

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

    const Op_t& op_;                // reference to operator object to use
    const int num_requested_;       // number of eigenpairs to calculate
    const int maxit_;               // maximum number of lanc interations
    const double tol_;              // convergence tolerance for eigenvalue
    const unsigned int percent_overshoot_;
                                    // allowed iteration count overshoot from
                                    // less frequent stopping tests
    const unsigned int verbosity_;  // verbosity of output

    size_t dim_;                    // operator and vector dimension
    size_t dim_tridiag_;            // dimension of tridiag matrix

    Vectors_t lanc_vectors_;        // lanczos vectors

    std::vector<Scalar_t> evals_;   // (approximate) eigenvals
    std::vector<Scalar_t> evecs_tridiag_;
                                    // eigenvecs of tridiag matrix,
                                    // stored in flat array

    Vectors_t evecs_;               // (approximate) eigenvecs of full operator

    void calc_tridiag_epairs_(int it, Scalars_t& t_maindiag,
                              Scalars_t& t_offdiag);

    bool check_convergence_(Scalar_t beta);

    void calc_evecs_();

    // Disallowed methods

    Lanczos(       const Lanczos<Op_t>&);
    void operator=(const Lanczos<Op_t>&);
};

} // namespace lanczos

} // namespace mfmg

#endif
