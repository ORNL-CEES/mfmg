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

#include <mfmg/cuda_matrix_operator.cuh>

#include <EpetraExt_MatrixMatrix.h>
#include <EpetraExt_Transpose_RowMatrix.h>

namespace mfmg
{
namespace
{
template <typename VectorType>
struct MatrixOperator
{
  static void
  apply(std::shared_ptr<SparseMatrixDevice<typename VectorType::value_type>>
            matrix,
        VectorType const &x, VectorType &y);
};

template <typename VectorType>
void MatrixOperator<VectorType>::apply(
    std::shared_ptr<SparseMatrixDevice<typename VectorType::value_type>> matrix,
    VectorType const &x, VectorType &y)
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <>
void MatrixOperator<VectorDevice<double>>::apply(
    std::shared_ptr<SparseMatrixDevice<double>> matrix,
    VectorDevice<double> const &x, VectorDevice<double> &y)
{
  matrix->vmult(y, x);
}

template <>
void MatrixOperator<dealii::LinearAlgebra::distributed::Vector<double>>::apply(
    std::shared_ptr<SparseMatrixDevice<double>> matrix,
    dealii::LinearAlgebra::distributed::Vector<double> const &x,
    dealii::LinearAlgebra::distributed::Vector<double> &y)
{
  // Move the data to the device
  VectorDevice<double> x_dev(x);
  VectorDevice<double> y_dev(y);

  matrix->vmult(y_dev, x_dev);

  // Move the data to the host
  std::vector<double> y_host(y.local_size());
  cuda_mem_copy_to_host(y_dev.val_dev, y_host);
  std::copy(y_host.begin(), y_host.end(), y.begin());
}
} // namespace

template <typename VectorType>
CudaMatrixOperator<VectorType>::CudaMatrixOperator(
    std::shared_ptr<SparseMatrixDevice<value_type>> sparse_matrix)
    : _matrix(sparse_matrix)
{
}

template <typename VectorType>
void CudaMatrixOperator<VectorType>::apply(VectorType const &x,
                                           VectorType &y) const
{
  MatrixOperator<VectorType>::apply(_matrix, x, y);
}

template <typename VectorType>
std::shared_ptr<Operator<VectorType>>
CudaMatrixOperator<VectorType>::transpose() const
{
  // Copy the data to the cpu and then, use trilinos to compute the
  // transpose. This is not the most efficient way to do this but it is the
  // easiest.
  auto sparse_matrix = convert_to_trilinos_matrix(*_matrix);

  // Transpose the sparse matrix
  auto epetra_matrix = sparse_matrix.trilinos_matrix();

  EpetraExt::RowMatrix_Transpose transposer;
  auto transposed_epetra_matrix =
      dynamic_cast<Epetra_CrsMatrix &>(transposer(epetra_matrix));

  auto transposed_matrix = std::make_shared<SparseMatrixDevice<double>>(
      convert_matrix(transposed_epetra_matrix));
  transposed_matrix->cusparse_handle = _matrix->cusparse_handle;
  cusparseStatus_t cusparse_error_code;
  cusparse_error_code = cusparseCreateMatDescr(&transposed_matrix->descr);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code = cusparseSetMatType(transposed_matrix->descr,
                                           CUSPARSE_MATRIX_TYPE_GENERAL);
  ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_error_code = cusparseSetMatIndexBase(transposed_matrix->descr,
                                                CUSPARSE_INDEX_BASE_ZERO);
  ASSERT_CUSPARSE(cusparse_error_code);

  return std::make_shared<CudaMatrixOperator<VectorType>>(transposed_matrix);
}

template <typename VectorType>
std::shared_ptr<Operator<VectorType>> CudaMatrixOperator<VectorType>::multiply(
    std::shared_ptr<Operator<VectorType> const> b) const
{
  // Downcast to SparseMatrixDeviceOperator
  auto downcast_b =
      std::dynamic_pointer_cast<CudaMatrixOperator<VectorType> const>(b);

  auto a_mat = this->get_matrix();
  auto b_mat = downcast_b->get_matrix();

  // Initialize c
  auto c_mat = std::make_shared<SparseMatrixDevice<double>>(*a_mat);

  a_mat->mmult(*c_mat, *b_mat);

  return std::make_shared<CudaMatrixOperator<VectorType>>(c_mat);
}

template <typename VectorType>
std::shared_ptr<Operator<VectorType>>
CudaMatrixOperator<VectorType>::multiply_transpose(
    std::shared_ptr<Operator<VectorType> const> b) const
{
  // Downcast operator
  auto downcast_b =
      std::dynamic_pointer_cast<CudaMatrixOperator<VectorType> const>(b);

  // TODO: to do this on the GPU
  // Copy the data to the cpu and then, use trilinos to compute the
  // transpose. This is not the most efficient way to do this but it is the
  // easiest.
  auto b_sparse_matrix_dev = downcast_b->get_matrix();
  auto b_sparse_matrix = convert_to_trilinos_matrix(*b_sparse_matrix_dev);

  // Transpose the sparse matrix
  auto b_epetra_matrix = b_sparse_matrix.trilinos_matrix();

  // In serial we can do the matrix-matrix multiplication on the device. In
  // parallel, we do everything on the host
  MPI_Comm comm = b_sparse_matrix_dev->get_mpi_communicator();
  unsigned int const comm_size = dealii::Utilities::MPI::n_mpi_processes(comm);
  if (comm_size == 1)
  {
    EpetraExt::RowMatrix_Transpose transposer;
    auto b_transposed_epetra_matrix =
        dynamic_cast<Epetra_CrsMatrix &>(transposer(b_epetra_matrix));
    auto b_transposed_matrix_dev = std::make_shared<SparseMatrixDevice<double>>(
        convert_matrix(b_transposed_epetra_matrix));
    b_transposed_matrix_dev->cusparse_handle = _matrix->cusparse_handle;
    cusparseStatus_t cusparse_error_code;
    cusparse_error_code =
        cusparseCreateMatDescr(&b_transposed_matrix_dev->descr);
    ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code = cusparseSetMatType(b_transposed_matrix_dev->descr,
                                             CUSPARSE_MATRIX_TYPE_GENERAL);
    ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_error_code = cusparseSetMatIndexBase(
        b_transposed_matrix_dev->descr, CUSPARSE_INDEX_BASE_ZERO);
    ASSERT_CUSPARSE(cusparse_error_code);

    // Perform the multiplication
    auto a = get_matrix();
    auto c = std::make_shared<SparseMatrixDevice<value_type>>(*a);
    a->mmult(*c, *b_transposed_matrix_dev);

    std::shared_ptr<Operator<VectorType>> op(
        new CudaMatrixOperator<VectorType>(c));

    return op;
  }
  else
  {
    auto a_sparse_matrix = convert_to_trilinos_matrix(*_matrix);
    auto a_epetra_matrix = a_sparse_matrix.trilinos_matrix();
    dealii::TrilinosWrappers::SparseMatrix c;
    int error_code = EpetraExt::MatrixMatrix::Multiply(
        a_epetra_matrix, false, b_epetra_matrix, true,
        const_cast<Epetra_CrsMatrix &>(c.trilinos_matrix()));
    ASSERT(error_code == 0,
           "EpetraExt::MatrixMatrix::Multiply() returned "
           "non-zero error code in "
           "DealIITrilinosMatrixOperator::multiply_transpose()");

    std::shared_ptr<Operator<VectorType>> op(new CudaMatrixOperator<VectorType>(
        std::make_shared<SparseMatrixDevice<value_type>>(convert_matrix(c))));

    return op;
  }
}

template <typename VectorType>
std::shared_ptr<VectorType>
CudaMatrixOperator<VectorType>::build_domain_vector() const
{
  auto partitioner =
      std::make_shared<const dealii::Utilities::MPI::Partitioner>(
          _matrix->locally_owned_domain_indices(),
          _matrix->get_mpi_communicator());

  return std::make_shared<vector_type>(partitioner);
}

template <typename VectorType>
std::shared_ptr<VectorType>
CudaMatrixOperator<VectorType>::build_range_vector() const
{
  auto partitioner =
      std::make_shared<const dealii::Utilities::MPI::Partitioner>(
          _matrix->locally_owned_range_indices(),
          _matrix->get_mpi_communicator());

  return std::make_shared<vector_type>(partitioner);
}

template <typename VectorType>
size_t CudaMatrixOperator<VectorType>::grid_complexity() const
{
  return _matrix->m();
}

template <typename VectorType>
size_t CudaMatrixOperator<VectorType>::operator_complexity() const
{
  return _matrix->n_nonzero_elements();
}

template <typename VectorType>
std::shared_ptr<SparseMatrixDevice<typename VectorType::value_type>>
CudaMatrixOperator<VectorType>::get_matrix() const
{
  return _matrix;
}
} // namespace mfmg

// Explicit Instantiation
template class mfmg::CudaMatrixOperator<mfmg::VectorDevice<double>>;
template class mfmg::CudaMatrixOperator<
    dealii::LinearAlgebra::distributed::Vector<double>>;
