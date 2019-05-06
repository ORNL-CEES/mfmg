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

#ifndef MFMG_CUDA_MATRIX_FREE_MESH_EVALUATOR_CUH
#define MFMG_CUDA_MATRIX_FREE_MESH_EVALUATOR_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/cuda/cuda_mesh_evaluator.cuh>

namespace mfmg
{
template <int dim>
class CudaMatrixFreeMeshEvaluator : public CudaMeshEvaluator<dim>
{
public:
  CudaMatrixFreeMeshEvaluator(CudaHandle const &cuda_handle,
                              dealii::DoFHandler<dim> &dof_handler,
                              dealii::AffineConstraints<double> &constraints)
      : CudaMeshEvaluator<dim>(cuda_handle, dof_handler, constraints)
  {
  }

  void apply(VectorDevice<double> const &src, VectorDevice<double> &dst) const;

  void apply(dealii::LinearAlgebra::distributed::Vector<double> const &src,
             dealii::LinearAlgebra::distributed::Vector<double> &dst) const;

  std::shared_ptr<dealii::LinearAlgebra::distributed::Vector<double>>
  build_range_vector() const;

  // TODO
  template <typename VectorType>
  VectorType get_diagonal_inverse() const
  {
    return VectorType();
  }
};
} // namespace mfmg

#endif

#endif
