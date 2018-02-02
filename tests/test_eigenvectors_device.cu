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

#define BOOST_TEST_MODULE eigenvectors_device

#include "main.cc"

#include <mfmg/amge_device.cuh>
#include <mfmg/sparse_matrix_device.cuh>
#include <mfmg/utils.cuh>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>

void diagonal_matrices(
    dealii::DoFHandler<2> &dof_handler,
    dealii::ConstraintMatrix &constraint_matrix,
    std::shared_ptr<mfmg::SparseMatrixDevice<double>> &system_matrix_dev,
    std::shared_ptr<mfmg::SparseMatrixDevice<double>> &mass_matrix_dev)
{
  // Build the matrix on the host
  dealii::FE_Q<2> fe(1);
  dof_handler.distribute_dofs(fe);
  constraint_matrix.clear();
  dealii::SparsityPattern system_sparsity_pattern;
  dealii::SparseMatrix<double> system_matrix;
  dealii::SparsityPattern mass_sparsity_pattern;
  dealii::SparseMatrix<double> mass_matrix;

  unsigned int const size = 30;
  std::vector<std::vector<unsigned int>> column_indices(
      size, std::vector<unsigned int>(1));
  for (unsigned int i = 0; i < size; ++i)
    column_indices[i][0] = i;
  mass_sparsity_pattern.copy_from(size, size, column_indices.begin(),
                                  column_indices.end());
  system_sparsity_pattern.copy_from(mass_sparsity_pattern);
  system_matrix.reinit(system_sparsity_pattern);
  mass_matrix.reinit(mass_sparsity_pattern);
  for (unsigned int i = 0; i < size; ++i)
  {
    system_matrix.diag_element(i) = static_cast<double>(i + 1);
    mass_matrix.diag_element(i) = 1.;
  }

  // Move the matrices to the device
  system_matrix_dev = std::make_shared<mfmg::SparseMatrixDevice<double>>(
      mfmg::convert_matrix(system_matrix));
  mass_matrix_dev = std::make_shared<mfmg::SparseMatrixDevice<double>>(
      mfmg::convert_matrix(mass_matrix));
  std::cout << system_matrix_dev.use_count() << std::endl;
  std::cout << system_matrix_dev->val_dev << std::endl;
}

BOOST_AUTO_TEST_CASE(diagonal)
{
  dealii::parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(3);
  dealii::FE_Q<2> fe(1);
  dealii::DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

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

  mfmg::AMGe_device<2, double> amge(MPI_COMM_WORLD, dof_handler,
                                    cusolver_dn_handle, cusparse_handle);

  unsigned int const n_eigenvalues = 5;
  std::map<typename dealii::Triangulation<2>::active_cell_iterator,
           typename dealii::DoFHandler<2>::active_cell_iterator>
      patch_to_global_map;
  for (auto cell : dof_handler.active_cell_iterators())
    patch_to_global_map[cell] = cell;

  double *eigenvalues_dev;
  double *eigenvectors_dev;
  std::vector<dealii::types::global_dof_index> dof_indices_map;
  std::tie(eigenvalues_dev, eigenvectors_dev, dof_indices_map) =
      amge.compute_local_eigenvectors(n_eigenvalues, triangulation,
                                      patch_to_global_map, diagonal_matrices);

  unsigned int const n_dofs = dof_handler.n_dofs();
  std::vector<dealii::types::global_dof_index> ref_dof_indices_map(n_dofs);
  std::iota(ref_dof_indices_map.begin(), ref_dof_indices_map.end(), 0);
  // We cannot use BOOST_TEST because it uses variadic template and there is
  // bug in CUDA 7.0 and CUDA 8.0 with variadic templates
  // See http://www.boost.org/doc/libs/1_66_0/boost/config/compiler/nvcc.hpp
  for (unsigned int i = 0; i < n_dofs; ++i)
    BOOST_CHECK_EQUAL(dof_indices_map[i], ref_dof_indices_map[i]);

  unsigned int const eigenvector_size = 30;
  std::vector<double> ref_eigenvalues(n_eigenvalues);
  std::vector<dealii::Vector<double>> ref_eigenvectors(
      n_eigenvalues, dealii::Vector<double>(eigenvector_size));
  for (unsigned int i = 0; i < n_eigenvalues; ++i)
  {
    ref_eigenvalues[i] = static_cast<double>(i + 1);
    ref_eigenvectors[i][i] = 1.;
  }

  cudaError_t cuda_error_code;
  for (unsigned int i = 0; i < n_eigenvalues; ++i)
  {
    std::vector<double> eigenvalues(n_eigenvalues);
    cuda_error_code =
        cudaMemcpy(&eigenvalues[0], eigenvalues_dev,
                   n_eigenvalues * sizeof(double), cudaMemcpyDeviceToHost);
    mfmg::ASSERT_CUDA(cuda_error_code);
    BOOST_CHECK_CLOSE(eigenvalues[i], ref_eigenvalues[i], 1e-12);

    std::vector<double> eigenvectors(n_eigenvalues * eigenvector_size);
    cuda_error_code =
        cudaMemcpy(&eigenvectors[0], eigenvectors_dev,
                   n_eigenvalues * eigenvector_size * sizeof(double),
                   cudaMemcpyDeviceToHost);
    mfmg::ASSERT_CUDA(cuda_error_code);
    for (unsigned int j = 0; j < eigenvector_size; ++j)
      BOOST_CHECK_CLOSE(std::abs(eigenvectors[i * eigenvector_size + j]),
                        ref_eigenvectors[i][j], 1e-12);
  }

  // Free memory allocated on device
  cuda_error_code = cudaFree(eigenvalues_dev);
  mfmg::ASSERT_CUDA(cuda_error_code);
  cuda_error_code = cudaFree(eigenvectors_dev);
  mfmg::ASSERT_CUDA(cuda_error_code);

  // Destroy cusolver_handle
  cusolver_error_code = cusolverDnDestroy(cusolver_dn_handle);
  mfmg::ASSERT_CUSOLVER(cusolver_error_code);
  cusolver_dn_handle = nullptr;

  // Destroy cusparse_handle
  cusparse_error_code = cusparseDestroy(cusparse_handle);
  mfmg::ASSERT_CUSPARSE(cusparse_error_code);
  cusparse_handle = nullptr;
}
