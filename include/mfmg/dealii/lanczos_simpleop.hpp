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

#ifndef _LANCZOS_SIMPLEOP_HPP_
#define _LANCZOS_SIMPLEOP_HPP_

#include <cstddef>
#include <vector>

namespace mfmg::lanczos
{

//-----------------------------------------------------------------------------

template<typename Vector_t_>
class SimpleOp {

  public:

    // Typedefs

    typedef Vector_t_ Vector_t;
    typedef typename Vector_t::Scalar_t Scalar_t;
    typedef typename std::vector<Vector_t*> Vectors_t;

    // Ctor/dtor

    SimpleOp(size_t dim, size_t multiplicity = 1);
    ~SimpleOp();

    // Accessors

    size_t dim() const {return dim_;}

    Scalar_t eigenvalue(size_t i) const {return diag_value_(i);}

    // Operations

    void apply(Vector_t& vout, const Vector_t& vin) const;

  private:

    size_t dim_;
    size_t multiplicity_;

    // This will be a diagonal matrix; specify the diag entries here

    Scalar_t diag_value_(size_t i) const {
      return (Scalar_t)(1+i/multiplicity_);
    }

    // Disallowed methods

    SimpleOp(      const SimpleOp<Vector_t>&);
    void operator=(const SimpleOp<Vector_t>&);
};

//-----------------------------------------------------------------------------

} // namespace mfmg::lanczos

#endif // _LANCZOS_SIMPLEOP_HPP_

//=============================================================================
