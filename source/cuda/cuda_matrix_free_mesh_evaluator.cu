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

#include <mfmg/cuda/cuda_matrix_free_mesh_evaluator.cuh>

namespace mfmg
{
template <int dim>
std::string CudaMatrixFreeMeshEvaluator<dim>::get_mesh_evaluator_type() const
{
  return "CudaMatrixFreeMeshEvaluator";
}

template <int dim>
std::shared_ptr<dealii::LinearAlgebra::distributed::Vector<double>>
CudaMatrixFreeMeshEvaluator<dim>::build_range_vector() const
{

  // Get the MPI communicator from the dof handler
  auto const &triangulation = (this->_dof_handler).get_triangulation();
  using Triangulation =
      typename std::remove_reference<decltype(triangulation)>::type;
  auto comm = static_cast<dealii::parallel::Triangulation<
      Triangulation::dimension, Triangulation::space_dimension> const &>(
                  triangulation)
                  .get_communicator();

  // Get the set of locally owned DoFs
  auto const &locally_owned_dofs = (this->_dof_handler).locally_owned_dofs();

  return std::make_shared<dealii::LinearAlgebra::distributed::Vector<double>>(
      locally_owned_dofs, comm);
}

template <int dim>
void CudaMatrixFreeMeshEvaluator<dim>::apply(
    dealii::LinearAlgebra::distributed::Vector<
        double, dealii::MemorySpace::CUDA> const &src,
    dealii::LinearAlgebra::distributed::Vector<
        double, dealii::MemorySpace::CUDA> &dst) const
{
  // TODO
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <int dim>
void CudaMatrixFreeMeshEvaluator<dim>::apply(
    dealii::LinearAlgebra::distributed::Vector<
        double, dealii::MemorySpace::Host> const &src,
    dealii::LinearAlgebra::distributed::Vector<
        double, dealii::MemorySpace::Host> &dst) const
{
  // TODO
  ASSERT_THROW_NOT_IMPLEMENTED();
}
} // namespace mfmg

// Explicit Instantiation
template class mfmg::CudaMatrixFreeMeshEvaluator<2>;
template class mfmg::CudaMatrixFreeMeshEvaluator<3>;
