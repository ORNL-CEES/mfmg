/*************************************************************************
 * Copyright (c) 2017-2018 by the mfmg authors                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#include <mfmg/dealii_smoother.hpp>
#include <mfmg/dealii_trilinos_matrix_operator.hpp>
#include <mfmg/exceptions.hpp>
#include <mfmg/instantiation.hpp>

namespace mfmg
{
template <typename VectorType>
DealIISmoother<VectorType>::DealIISmoother(
    std::shared_ptr<Operator<vector_type> const> op,
    std::shared_ptr<boost::property_tree::ptree const> params)
    : Smoother<VectorType>(op, params)
{
  std::string prec_name =
      this->_params->get("smoother.type", "Symmetric Gauss-Seidel");

  // We only do smoothing if the operator is a DealIITrilinosMatrixOperator
  auto trilinos_operator =
      std::dynamic_pointer_cast<DealIITrilinosMatrixOperator<VectorType> const>(
          this->_operator);
  auto sparse_matrix = trilinos_operator->get_matrix();

  std::transform(prec_name.begin(), prec_name.end(), prec_name.begin(),
                 ::tolower);
  if (prec_name == "symmetric gauss-seidel")
  {
    _smoother.reset(new dealii::TrilinosWrappers::PreconditionSSOR());
    static_cast<dealii::TrilinosWrappers::PreconditionSSOR *>(_smoother.get())
        ->initialize(*sparse_matrix);
  }
  else if (prec_name == "gauss-seidel")
  {
    _smoother.reset(new dealii::TrilinosWrappers::PreconditionSOR());
    static_cast<dealii::TrilinosWrappers::PreconditionSOR *>(_smoother.get())
        ->initialize(*sparse_matrix);
  }
  else if (prec_name == "jacobi")
  {
    _smoother.reset(new dealii::TrilinosWrappers::PreconditionJacobi());
    static_cast<dealii::TrilinosWrappers::PreconditionJacobi *>(_smoother.get())
        ->initialize(*sparse_matrix);
  }
  else if (prec_name == "ilu")
  {
    _smoother.reset(new dealii::TrilinosWrappers::PreconditionILU());
    static_cast<dealii::TrilinosWrappers::PreconditionILU *>(_smoother.get())
        ->initialize(*sparse_matrix);
  }
  else
    ASSERT_THROW(false, "Unknown smoother name: \"" + prec_name + "\"");
}

template <typename VectorType>
void DealIISmoother<VectorType>::apply(VectorType const &b, VectorType &x) const
{
  // r = -(b - Ax)
  vector_type r(b);
  this->_operator->apply(x, r);
  r.add(-1., b);

  // x = x + B^{-1} (-r)
  vector_type tmp(x);
  _smoother->vmult(tmp, r);
  x.add(-1., tmp);
}

} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_VECTORTYPE(TUPLE(DealIISmoother))
