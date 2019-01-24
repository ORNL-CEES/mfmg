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

#ifndef _LANCZOS_SIMPLEVECTOR_HPP_
#define _LANCZOS_SIMPLEVECTOR_HPP_

#include <cstddef>
#include <vector>

namespace mfmg::lanczos
{

//-----------------------------------------------------------------------------
/// \brief Simple vector
///
///        Implements a simple vector with elements stored in contiguous
///        memory locations in (CPU) memory.

template<typename Scalar_t_>
class SimpleVector {

  public:

    // Typedefs

    typedef Scalar_t_ Scalar_t;

    // Ctor/dtor

    SimpleVector(size_t dim);
    ~SimpleVector();

    // Accessors

    size_t dim() const {return dim_;}

    Scalar_t& elt(size_t i);
    Scalar_t const_elt(size_t i) const;

    // Operations

    void copy(const SimpleVector& x);
    void copy(const SimpleVector* x) {copy(*x);}

    void axpy(Scalar_t a, const SimpleVector& x);
    void axpy(Scalar_t a, const SimpleVector* x) {axpy(a, *x);}

    void scal(Scalar_t a);

    Scalar_t dot(const SimpleVector& x) const;
    Scalar_t dot(const SimpleVector* x) const {return dot(*x);}

    Scalar_t nrm2() const;

    void set_zero();

    void set_random(int seed = 0, double multiplier = 1,
                    double cmultiplier = 0);

    void print() const;

  private:

    size_t dim_;
    std::vector<Scalar_t> data_;

    // Disallowed methods

    SimpleVector(  const SimpleVector<Scalar_t>&);
    void operator=(const SimpleVector<Scalar_t>&);

};

//-----------------------------------------------------------------------------

} // namespace mfmg::lanczos

#endif // _LANCZOS_SIMPLEVECTOR_HPP_

//=============================================================================
