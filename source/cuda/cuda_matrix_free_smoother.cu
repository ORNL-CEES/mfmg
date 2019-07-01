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

#include <mfmg/cuda/cuda_matrix_free_operator.cuh>
#include <mfmg/cuda/cuda_matrix_free_smoother.cuh>
#include <mfmg/cuda/cuda_matrix_operator.cuh>
#include <mfmg/cuda/dealii_operator_device_helpers.cuh>
#include <mfmg/cuda/utils.cuh>

namespace mfmg
{
namespace
{
template <typename VectorType>
struct SmootherOperator
{
  static void
  apply(Operator<VectorType> const &op,
        SparseMatrixDevice<typename VectorType::value_type> const &smoother,
        VectorType const &b, VectorType &x);
};

template <typename VectorType>
void SmootherOperator<VectorType>::apply(
    Operator<VectorType> const &op,
    SparseMatrixDevice<typename VectorType::value_type> const &smoother,
    VectorType const &b, VectorType &x)
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <>
void SmootherOperator<dealii::LinearAlgebra::distributed::Vector<
    double, dealii::MemorySpace::CUDA>>::
    apply(Operator<dealii::LinearAlgebra::distributed::Vector<
              double, dealii::MemorySpace::CUDA>> const &op,
          SparseMatrixDevice<double> const &smoother,
          dealii::LinearAlgebra::distributed::Vector<
              double, dealii::MemorySpace::CUDA> const &b,
          dealii::LinearAlgebra::distributed::Vector<
              double, dealii::MemorySpace::CUDA> &x)
{
  // r = -(b - Ax)
  dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::CUDA>
      r(b);
  op.apply(x, r);
  r.add(-1., b);

  // x = x + B^{-1} (-r)
  dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::CUDA>
      tmp(x);
  smoother.vmult(tmp, r);
  x.add(-1., tmp);
}

template <typename ScalarType>
__global__ void extract_inv_diag(ScalarType const *const matrix_value,
                                 int const *const matrix_column_index,
                                 int const *const matrix_row_index,
                                 int const size, ScalarType *value)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
    if (matrix_column_index[idx] == matrix_row_index[idx])
      value[matrix_column_index[idx]] = 1. / matrix_value[idx];
}
} // namespace

template <int dim, typename VectorType>
CudaMatrixFreeSmoother<dim, VectorType>::CudaMatrixFreeSmoother(
    std::shared_ptr<Operator<VectorType> const> op,
    std::shared_ptr<boost::property_tree::ptree const> params)
    : Smoother<VectorType>(op, params)
{
  std::string prec_type = this->_params->get("smoother.type", "Jacobi");
  // Transform to lower case
  std::transform(prec_type.begin(), prec_type.end(), prec_type.begin(),
                 tolower);

  ASSERT_THROW(prec_type == "jacobi", "Only Jacobi smoother is implemented.");

  // Downcast the operator and ask the user for the inverse of the diagonal
  auto cuda_matrix_free_operator =
      std::dynamic_pointer_cast<CudaMatrixFreeOperator<dim, VectorType> const>(
          this->_operator);
  VectorType inv_diagonal =
      cuda_matrix_free_operator->get_diagonal_inverse()->get_vector();
  auto partitioner = inv_diagonal.get_partitioner();
  unsigned int const size = partitioner->local_size();
  value_type *val_dev = nullptr;
  cuda_malloc(val_dev, size);
  if (std::is_same<VectorType, dealii::LinearAlgebra::distributed::Vector<
                                   double, dealii::MemorySpace::CUDA>>::value)
  {
    cudaError_t cuda_error_code =
        cudaMemcpy(val_dev, inv_diagonal.get_values(), size * sizeof(double),
                   cudaMemcpyDeviceToDevice);
    ASSERT_CUDA(cuda_error_code);
  }
  else
  {
    cudaError_t cuda_error_code =
        cudaMemcpy(val_dev, inv_diagonal.get_values(), size * sizeof(double),
                   cudaMemcpyHostToDevice);
    ASSERT_CUDA(cuda_error_code);
  }

  int *column_index_dev = nullptr;
  cuda_malloc(column_index_dev, size);
  int *row_ptr_dev = nullptr;
  cuda_malloc(row_ptr_dev, size + 1);

  int n_blocks = 1 + (size - 1) / block_size;
  iota<<<n_blocks, block_size>>>(size, column_index_dev,
                                 inv_diagonal.local_range().first);

  n_blocks = 1 + size / block_size;
  iota<<<n_blocks, block_size>>>(size + 1, row_ptr_dev);

  _smoother.reinit(
      partitioner->get_mpi_communicator(), val_dev, column_index_dev,
      row_ptr_dev, size, partitioner->locally_owned_range(),
      partitioner->locally_owned_range(),
      cuda_matrix_free_operator->get_cuda_handle().cusparse_handle);
}

template <int dim, typename VectorType>
void CudaMatrixFreeSmoother<dim, VectorType>::apply(VectorType const &b,
                                                    VectorType &x) const
{
  SmootherOperator<VectorType>::apply(*this->_operator, _smoother, b, x);
}
} // namespace mfmg

template class mfmg::CudaMatrixFreeSmoother<
    2, dealii::LinearAlgebra::distributed::Vector<double,
                                                  dealii::MemorySpace::CUDA>>;
template class mfmg::CudaMatrixFreeSmoother<
    3, dealii::LinearAlgebra::distributed::Vector<double,
                                                  dealii::MemorySpace::CUDA>>;
