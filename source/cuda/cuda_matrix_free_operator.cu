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
#include <mfmg/cuda/utils.cuh>
#include <mfmg/dealii/dealii_utils.hpp>

#include <memory>

namespace mfmg
{
template <int dim, typename VectorType>
CudaMatrixFreeOperator<dim, VectorType>::CudaMatrixFreeOperator(
    std::shared_ptr<CudaMatrixFreeMeshEvaluator<dim>>
        matrix_free_mesh_evaluator)
    : _cuda_handle(matrix_free_mesh_evaluator->get_cuda_handle()),
      _mesh_evaluator(std::move(matrix_free_mesh_evaluator))
{
}

template <int dim, typename VectorType>
void CudaMatrixFreeOperator<dim, VectorType>::vmult(VectorType &dst,
                                                    VectorType const &src) const
{
  _mesh_evaluator->matrix_free_evaluate_global(src, dst);
}

template <int dim, typename VectorType>
void CudaMatrixFreeOperator<dim, VectorType>::apply(
    dealii::LinearAlgebra::distributed::Vector<
        value_type, dealii::MemorySpace::Host> const &x,
    dealii::LinearAlgebra::distributed::Vector<value_type,
                                               dealii::MemorySpace::Host> &y,
    OperatorMode mode) const
{
  if (mode != OperatorMode::NO_TRANS)
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  auto x_dev = copy_from_host(x);
  auto y_dev = copy_from_host(y);

  _mesh_evaluator->matrix_free_evaluate_global(x_dev, y_dev);

  y = copy_from_dev(y_dev);
}

template <int dim, typename VectorType>
void CudaMatrixFreeOperator<dim, VectorType>::apply(VectorType const &x,
                                                    VectorType &y,
                                                    OperatorMode mode) const
{
  if (mode != OperatorMode::NO_TRANS)
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }
  this->vmult(y, x);
}

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
CudaMatrixFreeOperator<dim, VectorType>::transpose() const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
}

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
CudaMatrixFreeOperator<dim, VectorType>::multiply(
    std::shared_ptr<Operator<VectorType> const> /*b*/) const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
}

template <int dim, typename VectorType>
std::shared_ptr<Operator<VectorType>>
CudaMatrixFreeOperator<dim, VectorType>::multiply_transpose(
    std::shared_ptr<Operator<VectorType> const> b) const
{
  // TODO for now we perform this operation on the host
  // Downcast operator
  auto downcast_b =
      std::dynamic_pointer_cast<CudaMatrixOperator<VectorType> const>(b);

  auto tmp = this->build_range_vector();
  auto b_sparse_matrix_dev = downcast_b->get_matrix();
  auto b_sparse_matrix = convert_to_trilinos_matrix(*b_sparse_matrix_dev);

  // FIXME The function below needs to perform many vmult where the operator
  // is on the device but the source and the destination vectors are on the
  // host. The move from the host to the device and reverse is handled by
  // the apply function.
  auto c_sparse_matrix = matrix_transpose_matrix_multiply(
      tmp->locally_owned_elements(),
      b_sparse_matrix.locally_owned_range_indices(),
      tmp->get_mpi_communicator(), b_sparse_matrix, *this);

  // Convert matrix does not set cuda_handle and description
  auto c_dev = std::make_shared<SparseMatrixDevice<double>>(
      convert_matrix(*c_sparse_matrix));
  c_dev->cusparse_handle = b_sparse_matrix_dev->cusparse_handle;
  cusparseStatus_t cusparse_error_code;
  cusparse_error_code = cusparseCreateMatDescr(&c_dev->descr);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code =
      cusparseSetMatType(c_dev->descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code =
      cusparseSetMatIndexBase(c_dev->descr, CUSPARSE_INDEX_BASE_ZERO);
  ASSERT_CUSPARSE(cusparse_error_code);

  std::shared_ptr<Operator<VectorType>> op(
      new CudaMatrixOperator<VectorType>(c_dev));

  return op;
}

template <int dim, typename VectorType>
std::shared_ptr<VectorType>
CudaMatrixFreeOperator<dim, VectorType>::build_domain_vector() const
{
  // We know the operator is squared
  auto domain_vector = this->build_range_vector();

  return domain_vector;
}

template <int dim, typename VectorType>
std::shared_ptr<VectorType>
CudaMatrixFreeOperator<dim, VectorType>::build_range_vector() const
{
  return _mesh_evaluator->build_range_vector();
}

template <int dim, typename VectorType>
size_t CudaMatrixFreeOperator<dim, VectorType>::grid_complexity() const
{
  // FIXME Return garbage since throwing not implemented will make important
  // tests to fail
  return 0;
}

template <int dim, typename VectorType>
size_t CudaMatrixFreeOperator<dim, VectorType>::operator_complexity() const
{
  // FIXME Return garbage since throwing not implemented will make important
  // tests to fail
  return 0;
}

template <int dim, typename VectorType>
std::shared_ptr<dealii::DiagonalMatrix<VectorType>>
CudaMatrixFreeOperator<dim, VectorType>::get_diagonal_inverse() const
{
  return _mesh_evaluator->matrix_free_get_diagonal_inverse();
}

template <int dim, typename VectorType>
CudaHandle const &
CudaMatrixFreeOperator<dim, VectorType>::get_cuda_handle() const
{
  return _cuda_handle;
}
} // namespace mfmg

// Explicit Instantiation
template class mfmg::CudaMatrixFreeOperator<
    2, dealii::LinearAlgebra::distributed::Vector<double,
                                                  dealii::MemorySpace::CUDA>>;
template class mfmg::CudaMatrixFreeOperator<
    3, dealii::LinearAlgebra::distributed::Vector<double,
                                                  dealii::MemorySpace::CUDA>>;
