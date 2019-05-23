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

#include <mfmg/cuda/amge_device.cuh>
#include <mfmg/cuda/cuda_matrix_free_hierarchy_helpers.cuh>
#include <mfmg/cuda/cuda_matrix_free_mesh_evaluator.cuh>
#include <mfmg/cuda/cuda_matrix_free_operator.cuh>
#include <mfmg/cuda/cuda_matrix_operator.cuh>

namespace mfmg
{
template <int dim, typename VectorType>
CudaMatrixFreeHierarchyHelpers<dim, VectorType>::CudaMatrixFreeHierarchyHelpers(
    CudaHandle const &cuda_handle)
    : CudaHierarchyHelpers<dim, VectorType>(cuda_handle)
{
}

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
CudaMatrixFreeHierarchyHelpers<dim, VectorType>::get_global_operator(
    std::shared_ptr<MeshEvaluator> mesh_evaluator)
{
  if (this->_operator == nullptr)
  {
    this->_operator.reset(
        new CudaMatrixFreeOperator<VectorType>(mesh_evaluator));
  }

  return this->_operator;
}

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
CudaMatrixFreeHierarchyHelpers<dim, VectorType>::build_restrictor(
    MPI_Comm comm, std::shared_ptr<MeshEvaluator> mesh_evaluator,
    std::shared_ptr<boost::property_tree::ptree const> params)
{
  // Downcast to CudaMatriFreeMeshEvaluator
  auto cuda_mesh_evaluator =
      std::dynamic_pointer_cast<CudaMatrixFreeMeshEvaluator<dim>>(
          mesh_evaluator);

  auto eigensolver_params = params->get_child("eigensolver");
  AMGe_device<dim, CudaMeshEvaluator<dim>, vector_type> amge(
      comm, cuda_mesh_evaluator->get_dof_handler(), this->_cuda_handle);

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
} // namespace mfmg

// Explicit Instantiation
template class mfmg::CudaMatrixFreeHierarchyHelpers<
    2, dealii::LinearAlgebra::distributed::Vector<double,
                                                  dealii::MemorySpace::CUDA>>;
template class mfmg::CudaMatrixFreeHierarchyHelpers<
    3, dealii::LinearAlgebra::distributed::Vector<double,
                                                  dealii::MemorySpace::CUDA>>;
template class mfmg::CudaMatrixFreeHierarchyHelpers<
    2, dealii::LinearAlgebra::distributed::Vector<double,
                                                  dealii::MemorySpace::Host>>;
template class mfmg::CudaMatrixFreeHierarchyHelpers<
    3, dealii::LinearAlgebra::distributed::Vector<double,
                                                  dealii::MemorySpace::Host>>;
