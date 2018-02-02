/*************************************************************************
 * Copyright (c) 2018 by the mfmg authors                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#define BOOST_TEST_MODULE restriction_device

#include "main.cc"

#include <mfmg/amge_device.cuh>
#include <mfmg/utils.cuh>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>

BOOST_AUTO_TEST_CASE(restriction_matrix_device)
{
  // Initialize cuSOLVER
  cusolverDnHandle_t cusolver_dn_handle = nullptr;
  cusolverStatus_t cusolver_error_code;
  cusolver_error_code = cusolverDnCreate(&cusolver_dn_handle);
  mfmg::ASSERT_CUSOLVER(cusolver_error_code);

  // Initialize cuSPARSE
  cusparseHandle_t cusparse_handle = nullptr;
  cusparseStatus_t cusparse_error_code;
  cusparse_error_code = cusparseCreate(&cusparse_handle);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);

  dealii::parallel::distributed::Triangulation<3> triangulation(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);
  dealii::FE_Q<3> fe(4);
  dealii::DoFHandler<3> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  mfmg::AMGe_device<3, float> amge(MPI_COMM_WORLD, dof_handler,
                                   cusolver_dn_handle, cusparse_handle);

  unsigned int const n_local_rows = 3;
  unsigned int const eigenvectors_size = 10;
  std::vector<double> eigenvectors(n_local_rows * eigenvectors_size);
  for (unsigned int i = 0; i < n_local_rows; ++i)
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
      eigenvectors[i * eigenvectors_size + j] =
          n_local_rows * eigenvectors_size + i * eigenvectors_size + j;

  std::vector<std::vector<dealii::types::global_dof_index>> dof_indices_maps(
      n_local_rows,
      std::vector<dealii::types::global_dof_index>(eigenvectors_size));
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, dof_handler.n_dofs() - 1);
  for (unsigned int i = 0; i < n_local_rows; ++i)
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
      dof_indices_maps[i][j] = distribution(generator);

  float *eigenvectors_dev = nullptr;
  cudaError_t cuda_error_code =
      cudaMalloc(&eigenvectors_dev, eigenvectors.size() * sizeof(float));
  mfmg::ASSERT_CUDA(cuda_error_code);
  cuda_error_code =
      cudaMemcpy(eigenvectors_dev, &eigenvectors[0],
                 eigenvectors.size() * sizeof(float), cudaMemcpyHostToDevice);
  mfmg::ASSERT_CUDA(cuda_error_code);

  mfmg::SparseMatrixDevice<float> restriction_matrix_dev =
      amge.compute_restriction_sparse_matrix(eigenvectors_dev,
                                             dof_indices_maps);

  // Check that the values in the restriction matrix are the same as the
  // eigenvectors
  BOOST_CHECK_EQUAL(restriction_matrix_dev.val_dev, eigenvectors_dev);

  std::vector<int> column_index(n_local_rows * eigenvectors_size);
  cuda_error_code =
      cudaMemcpy(&column_index[0], restriction_matrix_dev.column_index_dev,
                 column_index.size() * sizeof(int), cudaMemcpyDeviceToHost);
  mfmg::ASSERT_CUDA(cuda_error_code);

  std::vector<int> row_ptr(n_local_rows + 1);
  cuda_error_code =
      cudaMemcpy(&row_ptr[0], restriction_matrix_dev.row_ptr_dev,
                 (n_local_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  mfmg::ASSERT_CUDA(cuda_error_code);

  for (unsigned int i = 0; i < n_local_rows + 1; ++i)
    BOOST_CHECK_EQUAL(row_ptr[i], i * eigenvectors_size);

  for (unsigned int row = 0; row < n_local_rows; ++row)
    for (unsigned int col = 0; col < eigenvectors_size; ++col)
      BOOST_CHECK_EQUAL(column_index[row * eigenvectors_size + col],
                        dof_indices_maps[row][col]);

  // Destroy cusolver_handle
  cusolver_error_code = cusolverDnDestroy(cusolver_dn_handle);
  mfmg::ASSERT_CUSOLVER(cusolver_error_code);
  cusolver_dn_handle = nullptr;

  // Destroy cusparse_handle
  cusparse_error_code = cusparseDestroy(cusparse_handle);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_handle = nullptr;
}
