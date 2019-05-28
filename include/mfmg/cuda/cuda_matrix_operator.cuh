/**************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                            *
 * All rights reserved.                                                   *
 *                                                                        *
 * This file is part of the mfmg library. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the  *
 * top-level directory                                                    *
 *                                                                        *
 * SPDX-License-Identifier: BSD-3-Clause                                  *
 *************************************************************************/

#ifndef MFMG_CUDA_MATRIX_OPERATOR_CUH
#define MFMG_CUDA_MATRIX_OPERATOR_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/common/operator.hpp>
#include <mfmg/cuda/sparse_matrix_device.cuh>

namespace mfmg
{
template <typename VectorType>
class CudaMatrixOperator : public Operator<VectorType>
{
public:
  using value_type = typename VectorType::value_type;
  using vector_type = VectorType;

  CudaMatrixOperator(
      std::shared_ptr<SparseMatrixDevice<value_type>> sparse_matrix);

  virtual ~CudaMatrixOperator() override = default;

  void apply(vector_type const &x, vector_type &y,
             OperatorMode mode = OperatorMode::NO_TRANS) const override final;

  std::shared_ptr<Operator<VectorType>> transpose() const override final;

  std::shared_ptr<Operator<VectorType>>
  multiply(std::shared_ptr<Operator<VectorType> const> b) const override final;

  std::shared_ptr<Operator<vector_type>> multiply_transpose(
      std::shared_ptr<Operator<vector_type> const> b) const override final;

  std::shared_ptr<vector_type> build_domain_vector() const override final;

  std::shared_ptr<vector_type> build_range_vector() const override final;

  size_t grid_complexity() const override final;

  size_t operator_complexity() const override final;

  std::shared_ptr<SparseMatrixDevice<value_type>> get_matrix() const;

private:
  mutable std::shared_ptr<SparseMatrixDevice<value_type>> _matrix;
  mutable std::shared_ptr<SparseMatrixDevice<value_type>> _transposed_matrix;
};
} // namespace mfmg

#endif

#endif
