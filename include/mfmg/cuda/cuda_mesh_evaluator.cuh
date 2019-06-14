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

#ifndef MFMG_CUDA_MESH_EVALUATOR_CUH
#define MFMG_CUDA_MESH_EVALUATOR_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/common/exceptions.hpp>
#include <mfmg/common/mesh_evaluator.hpp>
#include <mfmg/cuda/cuda_handle.cuh>
#include <mfmg/cuda/sparse_matrix_device.cuh>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>

namespace mfmg
{
template <int dim>
class CudaMeshEvaluator : public MeshEvaluator
{
public:
  CudaMeshEvaluator(CudaHandle const &cuda_handle,
                    dealii::DoFHandler<dim> &dof_handler,
                    dealii::AffineConstraints<double> &constraints);

  virtual int get_dim() const override final;

  virtual std::string get_mesh_evaluator_type() const override;

  virtual void evaluate_agglomerate(dealii::DoFHandler<dim> &,
                                    dealii::AffineConstraints<double> &,
                                    SparseMatrixDevice<double> &) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  virtual void evaluate_global(dealii::DoFHandler<dim> &,
                               dealii::AffineConstraints<double> &,
                               SparseMatrixDevice<double> &) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  virtual dealii::LinearAlgebra::distributed::Vector<double,
                                                     dealii::MemorySpace::Host>
  get_locally_relevant_diag() const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return dealii::LinearAlgebra::distributed::Vector<double>();
  }

  CudaHandle const &get_cuda_handle() const;

  dealii::DoFHandler<dim> &get_dof_handler();

  dealii::AffineConstraints<double> &get_constraints();

protected:
  CudaHandle const &_cuda_handle;
  dealii::DoFHandler<dim> &_dof_handler;
  dealii::AffineConstraints<double> &_constraints;
};

} // namespace mfmg

#endif

#endif
