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

#include <mfmg/common/operator.hpp>
#include <mfmg/cuda/cuda_handle.cuh>
#include <mfmg/cuda/cuda_matrix_free_mesh_evaluator.cuh>

namespace mfmg
{
template <int dim, typename VectorType>
class CudaMatrixFreeOperator : public Operator<VectorType>
{
public:
  using size_type = typename VectorType::size_type;
  using value_type = typename VectorType::value_type;
  using vector_type = VectorType;

  CudaMatrixFreeOperator(std::shared_ptr<CudaMatrixFreeMeshEvaluator<dim>>
                             matrix_free_mesh_evaluator);

  void vmult(vector_type &dst, vector_type const &src) const;

  // Needed because the extract_row works only on the host
  void apply(dealii::LinearAlgebra::distributed::Vector<
                 value_type, dealii::MemorySpace::Host> const &x,
             dealii::LinearAlgebra::distributed::Vector<
                 value_type, dealii::MemorySpace::Host> &y,
             OperatorMode mode = OperatorMode::NO_TRANS) const;

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

  std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
  get_diagonal_inverse() const;

  CudaHandle const &get_cuda_handle() const;

private:
  CudaHandle const &_cuda_handle;
  std::shared_ptr<CudaMatrixFreeMeshEvaluator<dim>> _mesh_evaluator;
};
} // namespace mfmg

#endif

#endif
