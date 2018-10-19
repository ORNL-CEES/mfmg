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

#include <mfmg/sparse_matrix_device.cuh>
#include <mfmg/utils.cuh>
#include <mfmg/utils.hpp>

#include <deal.II/lac/trilinos_index_access.h>

namespace mfmg
{
namespace internal
{
template <typename ScalarType>
ScalarType *copy_to_gpu(std::vector<ScalarType> const &val)
{
  unsigned int const n_elements = val.size();
  ASSERT(n_elements > 0, "Cannot copy an empty vector to the device");
  ScalarType *val_dev;
  cudaError_t error_code =
      cudaMalloc(&val_dev, n_elements * sizeof(ScalarType));
  ASSERT_CUDA(error_code);
  error_code = cudaMemcpy(val_dev, &val[0], n_elements * sizeof(ScalarType),
                          cudaMemcpyHostToDevice);
  ASSERT_CUDA(error_code);

  return val_dev;
}
} // namespace internal

template <typename ScalarType>
SparseMatrixDevice<ScalarType>
convert_matrix(dealii::SparseMatrix<ScalarType> const &sparse_matrix)
{
  unsigned int const nnz = sparse_matrix.n_nonzero_elements();
  int const n_rows = sparse_matrix.m();
  int const row_ptr_size = n_rows + 1;
  std::vector<ScalarType> val;
  val.reserve(nnz);
  std::vector<int> column_index;
  column_index.reserve(nnz);
  std::vector<int> row_ptr(row_ptr_size, 0);

  for (int row = 0; row < n_rows; ++row)
  {
    auto p_end = sparse_matrix.end(row);
    unsigned int counter = 0;
    for (auto p = sparse_matrix.begin(row); p != p_end; ++p)
    {
      val.emplace_back(p->value());
      column_index.emplace_back(p->column());
      ++counter;
    }
    row_ptr[row + 1] = row_ptr[row] + counter;

    // If the matrix is square deal.II stores the diagonal first in each row so
    // we need to do some reordering
    if (sparse_matrix.m() == sparse_matrix.n())
    {
      // Sort the elements in the row
      unsigned int const offset = row_ptr[row];
      int const diag_index = column_index[offset];
      ScalarType diag_elem = sparse_matrix.diag_element(row);
      unsigned int pos = 1;
      while ((column_index[offset + pos] < row) && (pos < counter))
      {
        val[offset + pos - 1] = val[offset + pos];
        column_index[offset + pos - 1] = column_index[offset + pos];
        ++pos;
      }
      val[offset + pos - 1] = diag_elem;
      column_index[offset + pos - 1] = diag_index;
    }
  }

  return SparseMatrixDevice<ScalarType>(
      MPI_COMM_SELF, internal::copy_to_gpu(val),
      internal::copy_to_gpu(column_index), internal::copy_to_gpu(row_ptr), nnz,
      dealii::complete_index_set(n_rows),
      dealii::complete_index_set(sparse_matrix.n()));
}

SparseMatrixDevice<double>
convert_matrix(dealii::TrilinosWrappers::SparseMatrix const &sparse_matrix)
{
  unsigned int const n_local_rows = sparse_matrix.local_size();
  std::vector<double> val;
  std::vector<int> column_index;
  std::vector<int> row_ptr(n_local_rows + 1);
  unsigned int local_nnz = 0;
  for (unsigned int row = 0; row < n_local_rows; ++row)
  {
    int n_entries;
    double *values;
    int *indices;
    sparse_matrix.trilinos_matrix().ExtractMyRowView(row, n_entries, values,
                                                     indices);

    val.insert(val.end(), values, values + n_entries);
    row_ptr[row + 1] = row_ptr[row] + n_entries;
    // Trilinos does not store the column indices directly
    for (int i = 0; i < n_entries; ++i)
      column_index.push_back(dealii::TrilinosWrappers::global_column_index(
          sparse_matrix.trilinos_matrix(), indices[i]));
    local_nnz += n_entries;
  }

  return SparseMatrixDevice<double>(
      sparse_matrix.get_mpi_communicator(), internal::copy_to_gpu(val),
      internal::copy_to_gpu(column_index), internal::copy_to_gpu(row_ptr),
      local_nnz, sparse_matrix.locally_owned_range_indices(),
      sparse_matrix.locally_owned_domain_indices());
}

SparseMatrixDevice<double> convert_matrix(Epetra_CrsMatrix const &sparse_matrix)
{
  auto range_map = sparse_matrix.RangeMap();
  unsigned int const n_local_rows = range_map.NumMyElements();
  std::vector<int> row_gid(n_local_rows);
  range_map.MyGlobalElements(row_gid.data());
  dealii::IndexSet range_indexset(range_map.NumGlobalElements());
  range_indexset.add_indices(row_gid.begin(), row_gid.end());
  range_indexset.compress();

  auto domain_map = sparse_matrix.DomainMap();
  std::vector<int> column_gid(domain_map.NumMyElements());
  domain_map.MyGlobalElements(column_gid.data());
  dealii::IndexSet domain_indexset(domain_map.NumGlobalElements());
  domain_indexset.add_indices(column_gid.begin(), column_gid.end());
  domain_indexset.compress();

  unsigned int const local_nnz = sparse_matrix.NumMyNonzeros();
  int *row_ptr_host = nullptr;
  int *local_column_index_host = nullptr;
  double *val_host = nullptr;
  sparse_matrix.ExtractCrsDataPointers(row_ptr_host, local_column_index_host,
                                       val_host);
  std::vector<int> column_index(local_nnz);
  for (unsigned int i = 0; i < local_nnz; ++i)
    column_index[i] = sparse_matrix.GCID(local_column_index_host[i]);

  double *val_dev;
  cuda_malloc(val_dev, local_nnz);
  cudaError_t cuda_error;
  cuda_error = cudaMemcpy(val_dev, val_host, local_nnz * sizeof(double),
                          cudaMemcpyHostToDevice);
  ASSERT_CUDA(cuda_error);

  int *row_ptr_dev;
  unsigned int const row_ptr_size = n_local_rows + 1;
  cuda_malloc(row_ptr_dev, row_ptr_size);
  cuda_error = cudaMemcpy(row_ptr_dev, row_ptr_host, row_ptr_size * sizeof(int),
                          cudaMemcpyHostToDevice);
  ASSERT_CUDA(cuda_error);

  return SparseMatrixDevice<double>(
      dynamic_cast<Epetra_MpiComm const &>(sparse_matrix.Comm()).Comm(),
      val_dev, internal::copy_to_gpu(column_index), row_ptr_dev, local_nnz,
      range_indexset, domain_indexset);
}

dealii::TrilinosWrappers::SparseMatrix
convert_to_trilinos_matrix(SparseMatrixDevice<double> const &matrix_dev)
{
  unsigned int const local_nnz = matrix_dev.local_nnz();
  unsigned int const n_local_rows = matrix_dev.n_local_rows();
  std::vector<double> values(local_nnz);
  std::vector<int> column_index(local_nnz);
  std::vector<int> row_ptr(n_local_rows + 1);

  // Copy the data to the host
  cuda_mem_copy_to_host(matrix_dev.val_dev, values);
  cuda_mem_copy_to_host(matrix_dev.column_index_dev, column_index);
  cuda_mem_copy_to_host(matrix_dev.row_ptr_dev, row_ptr);

  // Create the sparse matrix on the host
  dealii::IndexSet locally_owned_rows =
      matrix_dev.locally_owned_range_indices();
  dealii::TrilinosWrappers::SparseMatrix sparse_matrix(
      locally_owned_rows, matrix_dev.locally_owned_domain_indices(),
      matrix_dev.get_mpi_communicator());

  unsigned int pos = 0;
  for (auto row : locally_owned_rows)
  {
    unsigned int const n_cols = row_ptr[pos + 1] - row_ptr[pos];
    sparse_matrix.set(
        row, n_cols,
        reinterpret_cast<unsigned int *>(column_index.data() + row_ptr[pos]),
        values.data() + row_ptr[pos]);
    ++pos;
  }
  sparse_matrix.compress(dealii::VectorOperation::insert);

  return sparse_matrix;
}

std::tuple<std::unordered_map<int, int>, std::unordered_map<int, int>>
csr_to_amgx(std::unordered_set<int> const &rows_sent,
            SparseMatrixDevice<double> &matrix_dev)
{
  unsigned int local_nnz = matrix_dev.local_nnz();
  int *row_index_coo_dev = nullptr;
  cuda_malloc(row_index_coo_dev, local_nnz);
  int n_local_rows = matrix_dev.n_local_rows();

  // Change to COO format. The only thing that needs to be change to go from CSR
  // to COO is to change row_ptr_dev with row_index_coo_dev.
  cusparseStatus_t cusparse_error_code = cusparseXcsr2coo(
      matrix_dev.cusparse_handle, matrix_dev.row_ptr_dev, local_nnz,
      n_local_rows, row_index_coo_dev, CUSPARSE_INDEX_BASE_ZERO);
  ASSERT_CUSPARSE(cusparse_error_code);

  // Move the values, the rows, and the columns to the host
  std::vector<double> value_host(local_nnz);
  cuda_mem_copy_to_host(matrix_dev.val_dev, value_host);
  std::vector<int> col_index_host(local_nnz);
  cuda_mem_copy_to_host(matrix_dev.column_index_dev, col_index_host);
  std::vector<int> row_index_host(local_nnz);
  cuda_mem_copy_to_host(row_index_coo_dev, row_index_host);

  // Renumber halo data behind the local data
  auto range_indexset = matrix_dev.locally_owned_range_indices();
  std::vector<unsigned int> global_rows;
  range_indexset.fill_index_vector(global_rows);
  std::unordered_map<int, int> halo_map;
  for (unsigned int i = 0; i < n_local_rows; ++i)
    halo_map[global_rows[i]] = i;
  unsigned int const n_rows = matrix_dev.m();
  int next_free_id = n_local_rows;
  dealii::IndexSet col_indexset(matrix_dev.n());
  col_indexset.add_indices(col_index_host.begin(), col_index_host.end());
  col_indexset.compress();
  for (auto index : col_indexset)
  {
    int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (range_indexset.is_element(index) == false)
    {
      halo_map[index] = next_free_id;
      ++next_free_id;
    }
  }
  for (auto &col_index : col_index_host)
    col_index = halo_map[col_index];

  // Reorder rows and columns. We need to move to the top the rows that are
  // locally owned
  int strictly_owned_rows = n_local_rows - rows_sent.size();
  std::unordered_map<int, int> local_map;
  next_free_id = strictly_owned_rows;
  int next_free_local_id = 0;
  for (unsigned int i = 0; i < n_local_rows; ++i)
  {
    if (rows_sent.count(i) != 1)
    {
      local_map[i] = next_free_local_id;
      ++next_free_local_id;
    }
    else
    {
      local_map[i] = next_free_id;
      ++next_free_id;
    }
  }

  for (auto &col_index : col_index_host)
  {
    if (col_index < n_local_rows)
      col_index = local_map[col_index];
  }

  for (auto &row_index : row_index_host)
    row_index = local_map[row_index];

  // Sort the vectors
  auto permutation = sort_permutation(row_index_host, col_index_host);
  apply_permutation_in_place(permutation, value_host);
  apply_permutation_in_place(permutation, col_index_host);
  apply_permutation_in_place(permutation, row_index_host);

  // Move the data back to the device
  cuda_mem_copy_to_dev(value_host, matrix_dev.val_dev);
  cuda_mem_copy_to_dev(col_index_host, matrix_dev.column_index_dev);
  cuda_mem_copy_to_dev(row_index_host, row_index_coo_dev);

  // Change to CSR format
  cusparse_error_code = cusparseXcoo2csr(
      matrix_dev.cusparse_handle, row_index_coo_dev, local_nnz, n_local_rows,
      matrix_dev.row_ptr_dev, CUSPARSE_INDEX_BASE_ZERO);

  // Free allocated memory
  cuda_free(row_index_coo_dev);

  return std::make_tuple(halo_map, local_map);
}

void all_gather(MPI_Comm communicator, unsigned int send_count,
                unsigned int *send_buffer, unsigned int recv_count,
                unsigned int *recv_buffer)
{
  int comm_size;
  MPI_Comm_size(communicator, &comm_size);
  // First gather the number of elements each proc will send
  std::vector<int> n_elem_per_procs(comm_size);
  MPI_Allgather(&send_count, 1, MPI_INT, n_elem_per_procs.data(), 1, MPI_INT,
                communicator);

  // Gather the elements
  std::vector<int> displs(comm_size);
  for (int i = 1; i < comm_size; ++i)
    displs[i] = displs[i - 1] + n_elem_per_procs[i - 1];
  MPI_Allgatherv(send_buffer, send_count, MPI_UNSIGNED, recv_buffer,
                 n_elem_per_procs.data(), displs.data(), MPI_UNSIGNED,
                 communicator);
}

void all_gather(MPI_Comm communicator, unsigned int send_count,
                float *send_buffer, unsigned int recv_count, float *recv_buffer)
{
  int comm_size;
  MPI_Comm_size(communicator, &comm_size);
  std::vector<int> n_elem_per_procs(comm_size);
  MPI_Allgather(&send_count, 1, MPI_INT, n_elem_per_procs.data(), 1, MPI_INT,
                communicator);

  // Gather the elements
  std::vector<int> displs(comm_size);
  for (int i = 1; i < comm_size; ++i)
    displs[i] = displs[i - 1] + n_elem_per_procs[i - 1];
  MPI_Allgatherv(send_buffer, send_count, MPI_FLOAT, recv_buffer,
                 n_elem_per_procs.data(), displs.data(), MPI_FLOAT,
                 communicator);
}

void all_gather(MPI_Comm communicator, unsigned int send_count,
                double *send_buffer, unsigned int recv_count,
                double *recv_buffer)
{
  // First gather the number of elements each proc will send
  int comm_size;
  MPI_Comm_size(communicator, &comm_size);
  std::vector<int> n_elem_per_procs(comm_size);
  MPI_Allgather(&send_count, 1, MPI_INT, n_elem_per_procs.data(), 1, MPI_INT,
                communicator);

  // Gather the elements
  std::vector<int> displs(comm_size);
  for (int i = 1; i < comm_size; ++i)
    displs[i] = displs[i - 1] + n_elem_per_procs[i - 1];
  MPI_Allgatherv(send_buffer, send_count, MPI_DOUBLE, recv_buffer,
                 n_elem_per_procs.data(), displs.data(), MPI_DOUBLE,
                 communicator);
}

#ifdef MFMG_WITH_CUDA_MPI
void all_gather_dev(MPI_Comm communicator, unsigned int send_count,
                    float *send_buffer, unsigned int recv_count,
                    float *recv_buffer)
{
  // First gather the number of elements each proc will send
  int comm_size;
  MPI_Comm_size(&communicator, &comm_size);
  if (comm_size > 1)
  {
    all_gather(communicator, send_count, send_buffer, recv_count, recv_buffer);
  }
  else
  {
    cudaError_t cuda_error_code;
    cuda_error_code =
        cudaMemcpy(recv_buffer, send_buffer, send_count * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    ASSERT_CUDA(cuda_error_code);
  }
}

void all_gather_dev(MPI_Comm communicator, unsigned int send_count,
                    double *send_buffer, unsigned int recv_count,
                    double *recv_buffer)
{
  // If there is only one proc, we just copy the value in the send_buffer to the
  // recv_buffer.
  int comm_size;
  MPI_Comm_size(communicator, &comm_size);
  if (comm_size > 1)
  {
    all_gather(communicator, send_count, send_buffer, recv_count, recv_buffer);
  }
  else
  {
    cudaError_t cuda_error_code;
    cuda_error_code =
        cudaMemcpy(recv_buffer, send_buffer, send_count * sizeof(double),
                   cudaMemcpyDeviceToDevice);
    ASSERT_CUDA(cuda_error_code);
  }
}
#else
void all_gather_dev(MPI_Comm communicator, unsigned int send_count,
                    float *send_buffer, unsigned int recv_count,
                    float *recv_buffer)
{
  // If there is only one proc, we just copy the value in the send_buffer to the
  // recv_buffer.
  int comm_size;
  MPI_Comm_size(communicator, &comm_size);
  if (comm_size > 1)
  {
    // We cannot call MPI directly, so first we copy the send_buffer to the host
    // and after the communication, we copy the result in the recv_buffer.
    std::vector<float> send_buffer_host(send_count);
    cudaError_t cuda_error_code;
    cuda_error_code =
        cudaMemcpy(&send_buffer_host[0], send_buffer,
                   send_count * sizeof(float), cudaMemcpyDeviceToHost);
    ASSERT_CUDA(cuda_error_code);
    std::vector<float> recv_buffer_host(recv_count);

    all_gather(communicator, send_count, &send_buffer_host[0], recv_count,
               &recv_buffer_host[0]);

    cuda_error_code =
        cudaMemcpy(recv_buffer, &recv_buffer_host[0],
                   recv_count * sizeof(float), cudaMemcpyHostToDevice);
    ASSERT_CUDA(cuda_error_code);
  }
  else
  {
    cudaError_t cuda_error_code;
    cuda_error_code =
        cudaMemcpy(recv_buffer, send_buffer, send_count * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    ASSERT_CUDA(cuda_error_code);
  }
}

void all_gather_dev(MPI_Comm communicator, unsigned int send_count,
                    double *send_buffer, unsigned int recv_count,
                    double *recv_buffer)
{
  // If there is only one proc, we just copy the value in the send_buffer to the
  // recv_buffer.
  int comm_size;
  MPI_Comm_size(communicator, &comm_size);
  if (comm_size > 1)
  {
    // We cannot call MPI directly, so first we copy the send_buffer to the host
    // and after the communication, we copy the result in the recv_buffer.
    std::vector<double> send_buffer_host(send_count);
    cudaError_t cuda_error_code;
    cuda_error_code =
        cudaMemcpy(&send_buffer_host[0], send_buffer,
                   send_count * sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_CUDA(cuda_error_code);
    std::vector<double> recv_buffer_host(recv_count);

    all_gather(communicator, send_count, &send_buffer_host[0], recv_count,
               &recv_buffer_host[0]);

    cuda_error_code =
        cudaMemcpy(recv_buffer, &recv_buffer_host[0],
                   recv_count * sizeof(double), cudaMemcpyHostToDevice);
    ASSERT_CUDA(cuda_error_code);
  }
  else
  {
    cudaError_t cuda_error_code;
    cuda_error_code =
        cudaMemcpy(recv_buffer, send_buffer, send_count * sizeof(double),
                   cudaMemcpyDeviceToDevice);
    ASSERT_CUDA(cuda_error_code);
  }
}
#endif

template SparseMatrixDevice<double>
convert_matrix(dealii::SparseMatrix<double> const &sparse_matrix);
} // namespace mfmg
