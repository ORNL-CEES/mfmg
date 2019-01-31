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

#ifndef MFMG_LANCZOS_DEFLATEDOP_HPP
#define MFMG_LANCZOS_DEFLATEDOP_HPP

#include <cstddef>
#include <vector>

namespace mfmg
{
namespace lanczos
{

//-----------------------------------------------------------------------------
/// \brief Deflated operator
///
///        Given an undeflated operator, a new operator is constructed
///        with a subspace represented by a set of vectors projected out.

template<typename BaseOp_t_>
class DeflatedOp {

  public:

    // Typedefs

    typedef BaseOp_t_ BaseOp_t;
    typedef typename BaseOp_t_::Vector_t Vector_t;
    typedef typename Vector_t::Scalar_t Scalar_t;
    typedef typename std::vector<Vector_t*> Vectors_t;

    // Ctor/dtor

    DeflatedOp(const BaseOp_t& base_op);
    ~DeflatedOp();

    // Accessors

    size_t dim() const {return dim_;}  // operator and vector dimension

    // Operations

    void apply(Vector_t& vout, const Vector_t& vin) const;

    void add_deflation_vecs(Vectors_t vecs);

    void deflate(Vector_t& vec) const;

  private:

    const BaseOp_t_& base_op_; // reference to the base operator object

    size_t dim_;  // operator and vector dimension

    Vectors_t deflation_vecs_;  // vectors to deflate out

    // Disallowed methods

    DeflatedOp(    const DeflatedOp<BaseOp_t>&);
    void operator=(const DeflatedOp<BaseOp_t>&);
};

} // namespace lanczos

} // namespace mfmg

#endif
