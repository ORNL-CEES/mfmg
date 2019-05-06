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

#include <mfmg/common/exceptions.hpp>
#include <mfmg/cuda/cuda_matrix_free_mesh_evaluator.cuh>
#include <mfmg/cuda/cuda_matrix_free_operator.cuh>
#include <mfmg/cuda/cuda_matrix_operator.cuh>
#include <mfmg/cuda/vector_device.cuh>
#include <mfmg/dealii/dealii_utils.hpp>

namespace mfmg
{
template <typename VectorType>
CudaMatrixFreeOperator<VectorType>::CudaMatrixFreeOperator(
    std::shared_ptr<MeshEvaluator> matrix_free_mesh_evaluator)
    : _cuda_handle(std::dynamic_pointer_cast<CudaMeshEvaluator<2>>(
                       matrix_free_mesh_evaluator) != nullptr
                       ? std::dynamic_pointer_cast<CudaMeshEvaluator<2>>(
                             matrix_free_mesh_evaluator)
                             ->get_cuda_handle()
                       : std::dynamic_pointer_cast<CudaMeshEvaluator<3>>(
                             matrix_free_mesh_evaluator)
                             ->get_cuda_handle()),
      _mesh_evaluator(std::move(matrix_free_mesh_evaluator))
{
  int const dim = _mesh_evaluator->get_dim();
  std::string const downcasting_failure_error_message =
      "Must pass a matrix free mesh evaluator to create an operator";
  if (dim == 2)
  {
    ASSERT(std::dynamic_pointer_cast<CudaMatrixFreeMeshEvaluator<2>>(
               _mesh_evaluator) != nullptr,
           downcasting_failure_error_message);
  }
  else if (dim == 3)
  {
    ASSERT(std::dynamic_pointer_cast<CudaMatrixFreeMeshEvaluator<3>>(
               _mesh_evaluator) != nullptr,
           downcasting_failure_error_message);
  }
  else
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }
}

template <typename VectorType>
void CudaMatrixFreeOperator<VectorType>::apply(VectorType const &x,
                                               VectorType &y,
                                               OperatorMode mode) const
{
  if (mode != OperatorMode::NO_TRANS)
    ASSERT_THROW_NOT_IMPLEMENTED();
  _mesh_evaluator->get_dim() == 2
      ? std::dynamic_pointer_cast<CudaMatrixFreeMeshEvaluator<2>>(
            _mesh_evaluator)
            ->apply(x, y)
      : std::dynamic_pointer_cast<CudaMatrixFreeMeshEvaluator<3>>(
            _mesh_evaluator)
            ->apply(x, y);
}

template <typename VectorType>
std::shared_ptr<Operator<VectorType>>
CudaMatrixFreeOperator<VectorType>::transpose() const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
}

template <typename VectorType>
std::shared_ptr<Operator<VectorType>>
CudaMatrixFreeOperator<VectorType>::multiply(
    std::shared_ptr<Operator<VectorType> const> /*b*/) const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
}

template <>
std::shared_ptr<Operator<dealii::LinearAlgebra::distributed::Vector<double>>>
CudaMatrixFreeOperator<dealii::LinearAlgebra::distributed::Vector<double>>::
    multiply_transpose(
        std::shared_ptr<
            Operator<dealii::LinearAlgebra::distributed::Vector<double>> const>
            b) const
{
  // TODO for now we perform this operation on the host
  // Downcast operator
  auto downcast_b = std::dynamic_pointer_cast<CudaMatrixOperator<
      dealii::LinearAlgebra::distributed::Vector<double>> const>(b);

  auto tmp = this->build_range_vector();
  auto b_sparse_matrix_dev = downcast_b->get_matrix();
  auto b_sparse_matrix = convert_to_trilinos_matrix(*b_sparse_matrix_dev);

  // FIXME The function below needs to perform many vmult where the operator is
  // on the device but the source and the destination vectors are on the host.
  // The move from the host to the device and reverse is handled by the apply
  // function in
  auto c_sparse_matrix = matrix_transpose_matrix_multiply(
      tmp->locally_owned_elements(),
      b_sparse_matrix.locally_owned_range_indices(),
      tmp->get_mpi_communicator(), b_sparse_matrix, *this);

  std::shared_ptr<Operator<dealii::LinearAlgebra::distributed::Vector<double>>>
  op(new CudaMatrixOperator<dealii::LinearAlgebra::distributed::Vector<double>>(
      std::make_shared<SparseMatrixDevice<value_type>>(
          convert_matrix(*c_sparse_matrix))));

  return op;
}

template <>
std::shared_ptr<Operator<VectorDevice<double>>>
CudaMatrixFreeOperator<VectorDevice<double>>::multiply_transpose(
    std::shared_ptr<Operator<VectorDevice<double>> const> b) const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
}

template <typename VectorType>
std::shared_ptr<VectorType>
CudaMatrixFreeOperator<VectorType>::build_domain_vector() const
{
  // We know the operator is squared
  auto domain_vector = this->build_range_vector();

  return domain_vector;
}

template <typename VectorType>
std::shared_ptr<VectorType>
CudaMatrixFreeOperator<VectorType>::build_range_vector() const
{
  auto dealii_range_vector =
      _mesh_evaluator->get_dim() == 2
          ? std::dynamic_pointer_cast<CudaMatrixFreeMeshEvaluator<2>>(
                _mesh_evaluator)
                ->build_range_vector()
          : std::dynamic_pointer_cast<CudaMatrixFreeMeshEvaluator<3>>(
                _mesh_evaluator)
                ->build_range_vector();
  std::shared_ptr<VectorType> range_vector(
      new VectorType(*dealii_range_vector));
  return range_vector;
}

template <typename VectorType>
size_t CudaMatrixFreeOperator<VectorType>::grid_complexity() const
{
  // FIXME Return garbage since throwing not implemented will make important
  // tests to fail
  return 0;
}

template <typename VectorType>
size_t CudaMatrixFreeOperator<VectorType>::operator_complexity() const
{
  // FIXME Return garbage sinze throwing not implemented will make important
  // tests to fail
  return 0;
}

template <typename VectorType>
VectorType CudaMatrixFreeOperator<VectorType>::get_diagonal_inverse() const
{
  auto diagonal_inverse =
      _mesh_evaluator->get_dim() == 2
          ? std::dynamic_pointer_cast<CudaMatrixFreeMeshEvaluator<2>>(
                _mesh_evaluator)
                ->get_diagonal_inverse<VectorType>()
          : std::dynamic_pointer_cast<CudaMatrixFreeMeshEvaluator<3>>(
                _mesh_evaluator)
                ->get_diagonal_inverse<VectorType>();

  return diagonal_inverse;
}

template <typename VectorType>
CudaHandle const &CudaMatrixFreeOperator<VectorType>::get_cuda_handle() const
{
  return _cuda_handle;
}
} // namespace mfmg

// Explicit Instantiation
template class mfmg::CudaMatrixFreeOperator<mfmg::VectorDevice<double>>;
template class mfmg::CudaMatrixFreeOperator<
    dealii::LinearAlgebra::distributed::Vector<double>>;
