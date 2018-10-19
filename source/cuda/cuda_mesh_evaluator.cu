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

#include <mfmg/cuda/cuda_mesh_evaluator.cuh>

namespace mfmg
{
template <int dim>
CudaMeshEvaluator<dim>::CudaMeshEvaluator(
    CudaHandle const &cuda_handle, dealii::DoFHandler<dim> &dof_handler,
    dealii::AffineConstraints<double> &constraints)
    : _cuda_handle(cuda_handle), _dof_handler(dof_handler),
      _constraints(constraints)
{
}

template <int dim>
int CudaMeshEvaluator<dim>::get_dim() const
{
  return dim;
}

template <int dim>
std::string CudaMeshEvaluator<dim>::get_mesh_evaluator_type() const
{
  return "CudaMeshEvaluator";
}

template <int dim>
CudaHandle const &CudaMeshEvaluator<dim>::get_cuda_handle() const
{
  return _cuda_handle;
}

template <int dim>
dealii::DoFHandler<dim> &CudaMeshEvaluator<dim>::get_dof_handler()
{
  return _dof_handler;
}

template <int dim>
dealii::AffineConstraints<double> &CudaMeshEvaluator<dim>::get_constraints()
{
  return _constraints;
}
} // namespace mfmg

// Explicit Instantiation
template class mfmg::CudaMeshEvaluator<2>;
template class mfmg::CudaMeshEvaluator<3>;
