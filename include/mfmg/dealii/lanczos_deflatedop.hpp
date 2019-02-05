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

#include <mfmg/common/exceptions.hpp>
#include <mfmg/common/operator.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace mfmg
{

//-----------------------------------------------------------------------------
/// \brief Deflated operator
///
///        Given an undeflated operator, a new operator is constructed
///        with a subspace represented by a set of vectors projected out.

template <typename VectorType>
class DeflatedOperator : public Operator<VectorType>
{

public:
  // Typedefs
  using vector_type = VectorType;
  using ScalarType = typename VectorType::value_type;

  // Ctor/dtor

  DeflatedOperator(const Operator<VectorType> &op);
  ~DeflatedOperator() {}

  // Operations

  void apply(vector_type const &x, vector_type &y,
             OperatorMode mode = OperatorMode::NO_TRANS) const;

  std::shared_ptr<vector_type> build_domain_vector() const
  {
    return _base_op.build_domain_vector();
  }

  std::shared_ptr<vector_type> build_range_vector() const
  {
    return _base_op.build_range_vector();
  }

  void add_deflation_vecs(std::vector<VectorType> const &vecs);

  void deflate(VectorType &vec) const;

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
  const Operator<VectorType> &_base_op; // reference to the base operator object

  std::vector<VectorType> _deflation_vecs; // vectors to deflate out

  // Disallowed methods

  DeflatedOperator(const DeflatedOperator<VectorType> &);
  void operator=(const DeflatedOperator<VectorType> &);
};

} // namespace mfmg

#endif
