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

#ifndef MFMG_CUDA_MATRIX_FREE_OPERATOR_CUH
#define MFMG_CUDA_MATRIX_FREE_OPERATOR_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/common/mesh_evaluator.hpp>
#include <mfmg/common/operator.hpp>
#include <mfmg/cuda/cuda_handle.cuh>

namespace mfmg
{
template <typename VectorType>
class CudaMatrixFreeOperator : public Operator<VectorType>
{
public:
  using size_type = typename VectorType::size_type;
  using value_type = typename VectorType::value_type;
  using vector_type = VectorType;

  CudaMatrixFreeOperator(
      std::shared_ptr<MeshEvaluator> matrix_free_mesh_evaluator);

  void apply(vector_type const &x, vector_type &y,
             OperatorMode mode = OperatorMode::NO_TRANS) const override final;

  std::shared_ptr<Operator<vector_type>> transpose() const override final;

  std::shared_ptr<Operator<vector_type>>
  multiply(std::shared_ptr<Operator<vector_type> const> b) const override final;

  std::shared_ptr<Operator<vector_type>> multiply_transpose(
      std::shared_ptr<Operator<vector_type> const> b) const override final;

  std::shared_ptr<vector_type> build_domain_vector() const override final;

  std::shared_ptr<vector_type> build_range_vector() const override final;

  size_t grid_complexity() const override final;

  size_t operator_complexity() const override final;

  VectorType get_diagonal_inverse() const;

  CudaHandle const &get_cuda_handle() const;

private:
  CudaHandle const &_cuda_handle;
  std::shared_ptr<MeshEvaluator> _mesh_evaluator;
};
} // namespace mfmg

#endif

#endif
