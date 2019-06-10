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

#include <deal.II/lac/read_write_vector.h>

#include <random>

namespace mfmg
{
template <int dim>
std::string CudaMatrixFreeMeshEvaluator<dim>::get_mesh_evaluator_type() const
{
  return "CudaMatrixFreeMeshEvaluator";
}

template <int dim>
std::shared_ptr<dealii::LinearAlgebra::distributed::Vector<
    double, dealii::MemorySpace::CUDA>>
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

  return std::make_shared<dealii::LinearAlgebra::distributed::Vector<
      double, dealii::MemorySpace::CUDA>>(locally_owned_dofs, comm);
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

template <int dim>
void CudaMatrixFreeMeshEvaluator<dim>::set_initial_guess(
    dealii::AffineConstraints<double> &constraints,
    dealii::LinearAlgebra::distributed::Vector<
        double, dealii::MemorySpace::CUDA> &x) const
{
  // The vector is of distributed type but we know that it is in fact serial
  unsigned int const n = x.size();

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0., 1.);
  dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(n);
  for (unsigned int i = 0; i < n; ++i)
  {
    rw_vector[i] =
        (!constraints.is_constrained(i) ? distribution(generator) : 0.);
  }

  x.import(rw_vector, dealii::VectorOperation::insert);
}

} // namespace mfmg

// Explicit Instantiation
template class mfmg::CudaMatrixFreeMeshEvaluator<2>;
template class mfmg::CudaMatrixFreeMeshEvaluator<3>;
