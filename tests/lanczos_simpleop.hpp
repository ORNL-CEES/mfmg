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

#ifndef MFMG_LANCZOS_SIMPLEOP_HPP
#define MFMG_LANCZOS_SIMPLEOP_HPP

#include <mfmg/common/exceptions.hpp>
#include <mfmg/common/operator.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace mfmg
{

//-----------------------------------------------------------------------------
/// \brief Simple test operator
///
///        A diagonal matrix with equally spaced eigenvalues of some
///        multiplicity.

template <typename VectorType>
class SimpleOperator : public Operator<VectorType>
{
public:
  // Typedefs
  using vector_type = VectorType;
  using ScalarType = typename VectorType::value_type;

  typedef typename std::vector<VectorType *> Vectors_t;

  // Ctor/dtor

  SimpleOperator(size_t dim, size_t multiplicity = 1);
  ~SimpleOperator();

  // Accessors
  ScalarType eigenvalue(size_t i) const { return diag_value_(i); }

  // Operations

  void apply(VectorType const &vin, VectorType &vout,
             OperatorMode mode = OperatorMode::NO_TRANS) const;

  std::shared_ptr<vector_type> build_domain_vector() const
  {
    return std::make_shared<vector_type>(_dim);
  }

  std::shared_ptr<vector_type> build_range_vector() const
  {
    return std::make_shared<vector_type>(_dim);
  }

  // Not implemented functions from Operator
  std::shared_ptr<Operator<VectorType>> transpose() const
  {
    ASSERT(true, "Not implemented");
    return nullptr;
  }

  std::shared_ptr<Operator<VectorType>>
  multiply(std::shared_ptr<Operator<VectorType> const>) const
  {
    ASSERT(true, "Not implemented");
    return nullptr;
    ;
  }

  std::shared_ptr<Operator<VectorType>>
  multiply_transpose(std::shared_ptr<Operator<VectorType> const>) const
  {
    ASSERT(true, "Not implemented");
    return nullptr;
    ;
  }

  size_t grid_complexity() const
  {
    ASSERT(true, "Not implemented");
    return 0;
  }

  size_t operator_complexity() const
  {
    ASSERT(true, "Not implemented");
    return 0;
  }

private:
  size_t _dim;
  size_t _multiplicity;

  // This will be a diagonal matrix; specify the diag entries here

  ScalarType diag_value_(size_t i) const
  {
    return (ScalarType)(1 + i / _multiplicity);
  }

  // Disallowed methods

  SimpleOperator(const SimpleOperator<VectorType> &);
  void operator=(const SimpleOperator<VectorType> &);
}; // namespace mfmg

} // namespace mfmg

#endif
