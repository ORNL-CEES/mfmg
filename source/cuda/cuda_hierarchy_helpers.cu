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

#include <mfmg/cuda/amge_device.cuh>
#include <mfmg/cuda/cuda_hierarchy_helpers.cuh>
#include <mfmg/cuda/cuda_matrix_operator.cuh>
#include <mfmg/cuda/cuda_mesh_evaluator.cuh>
#include <mfmg/cuda/cuda_smoother.cuh>
#include <mfmg/cuda/cuda_solver.cuh>
#include <mfmg/cuda/sparse_matrix_device.cuh>

namespace mfmg
{
template <int dim, typename VectorType>
CudaHierarchyHelpers<dim, VectorType>::CudaHierarchyHelpers(
    CudaHandle const &cuda_handle)
    : _cuda_handle(cuda_handle)
{
}

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
CudaHierarchyHelpers<dim, VectorType>::get_global_operator(
    std::shared_ptr<MeshEvaluator> mesh_evaluator)
{
  if (_operator == nullptr)
  {
    // Downcast to CudaMeshEvaluator
    std::shared_ptr<CudaMeshEvaluator<dim>> cuda_mesh_evaluator =
        std::dynamic_pointer_cast<CudaMeshEvaluator<dim>>(mesh_evaluator);

    auto system_matrix =
        std::make_shared<SparseMatrixDevice<typename VectorType::value_type>>();

    // Call user function to fill in the system matrix
    cuda_mesh_evaluator->evaluate_global(cuda_mesh_evaluator->get_dof_handler(),
                                         cuda_mesh_evaluator->get_constraints(),
                                         *system_matrix);

    _operator.reset(new CudaMatrixOperator<VectorType>(system_matrix));
  }

  return _operator;
}

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
CudaHierarchyHelpers<dim, VectorType>::build_restrictor(
    MPI_Comm comm, std::shared_ptr<MeshEvaluator> mesh_evaluator,
    std::shared_ptr<boost::property_tree::ptree const> params)
{
  // Downcast to CudaMeshEvaluator
  std::shared_ptr<CudaMeshEvaluator<dim>> cuda_mesh_evaluator =
      std::dynamic_pointer_cast<CudaMeshEvaluator<dim>>(mesh_evaluator);

  auto eigensolver_params = params->get_child("eigensolver");
  AMGe_device<dim, CudaMeshEvaluator<dim>, vector_type> amge(
      comm, cuda_mesh_evaluator->get_dof_handler(), _cuda_handle);

  auto agglomerate_params = params->get_child("agglomeration");
  int n_eigenvectors = eigensolver_params.get("number of eigenvectors", 1);
  double tolerance = eigensolver_params.get("tolerance", 1e-14);

  auto restrictor_matrix =
      std::make_shared<SparseMatrixDevice<typename VectorType::value_type>>(
          amge.setup_restrictor(agglomerate_params, n_eigenvectors, tolerance,
                                *cuda_mesh_evaluator));

  auto restrictor =
      std::make_shared<CudaMatrixOperator<VectorType>>(restrictor_matrix);

  return restrictor;
}

template <int dim, typename VectorType>
std::shared_ptr<Smoother<VectorType>>
CudaHierarchyHelpers<dim, VectorType>::build_smoother(
    std::shared_ptr<Operator<VectorType> const> op,
    std::shared_ptr<boost::property_tree::ptree const> params)
{
  return std::make_shared<CudaSmoother<VectorType>>(op, params);
}

template <int dim, typename VectorType>
std::shared_ptr<Solver<VectorType>>
CudaHierarchyHelpers<dim, VectorType>::build_coarse_solver(
    std::shared_ptr<Operator<VectorType> const> op,
    std::shared_ptr<boost::property_tree::ptree const> params)
{

  auto cuda_solver =
      std::make_shared<CudaSolver<VectorType>>(_cuda_handle, op, params);

  return cuda_solver;
}
} // namespace mfmg

// Explicit Instantiation
template class mfmg::CudaHierarchyHelpers<2, mfmg::VectorDevice<double>>;
template class mfmg::CudaHierarchyHelpers<3, mfmg::VectorDevice<double>>;
template class mfmg::CudaHierarchyHelpers<
    2, dealii::LinearAlgebra::distributed::Vector<double>>;
template class mfmg::CudaHierarchyHelpers<
    3, dealii::LinearAlgebra::distributed::Vector<double>>;
