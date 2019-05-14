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

#define BOOST_TEST_MODULE eigenvectors_device

#include <mfmg/cuda/amge_device.cuh>
#include <mfmg/cuda/cuda_mesh_evaluator.cuh>
#include <mfmg/cuda/sparse_matrix_device.cuh>
#include <mfmg/cuda/utils.cuh>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "main.cc"

template <int dim>
class DiagonalTestMeshEvaluator : public mfmg::CudaMeshEvaluator<dim>
{
public:
  DiagonalTestMeshEvaluator(mfmg::CudaHandle const &cuda_handle,
                            dealii::DoFHandler<dim> &dof_handler,
                            dealii::AffineConstraints<double> &constraints)
      : mfmg::CudaMeshEvaluator<dim>(cuda_handle, dof_handler, constraints)
  {
  }

  void evaluate_agglomerate(
      dealii::DoFHandler<2> &dof_handler,
      dealii::AffineConstraints<double> &constraint_matrix,
      mfmg::SparseMatrixDevice<double> &system_matrix_dev) const override final
  {
    // Build the matrix on the host
    dealii::FE_Q<2> fe(1);
    dof_handler.distribute_dofs(fe);
    constraint_matrix.clear();
    dealii::SparsityPattern system_sparsity_pattern;
    dealii::SparseMatrix<double> system_matrix;

    unsigned int const size = 30;
    std::vector<std::vector<unsigned int>> column_indices(
        size, std::vector<unsigned int>(1));
    for (unsigned int i = 0; i < size; ++i)
      column_indices[i][0] = i;
    system_sparsity_pattern.copy_from(size, size, column_indices.begin(),
                                      column_indices.end());
    system_matrix.reinit(system_sparsity_pattern);
    for (unsigned int i = 0; i < size; ++i)
    {
      system_matrix.diag_element(i) = static_cast<double>(i + 1);
    }

    // Move the matrices to the device
    system_matrix_dev = std::move(mfmg::convert_matrix(system_matrix));
  }

  void evaluate_global(
      dealii::DoFHandler<2> &dof_handler,
      dealii::AffineConstraints<double> &constraint_matrix,
      mfmg::SparseMatrixDevice<double> &system_matrix_dev) const override final
  {
  }
};

BOOST_AUTO_TEST_CASE(diagonal)
{
  int constexpr dim = 2;
  using Vector = dealii::LinearAlgebra::distributed::Vector<double>;
  using DummyMeshEvaluator = mfmg::CudaMeshEvaluator<dim>;

  dealii::parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(3);
  dealii::FE_Q<dim> fe(1);
  dealii::DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  // Initialize the CUDA libraries
  mfmg::CudaHandle cuda_handle;

  mfmg::AMGe_device<dim, DummyMeshEvaluator, Vector> amge(
      MPI_COMM_WORLD, dof_handler, cuda_handle);

  unsigned int const n_eigenvectors = 5;
  std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
           typename dealii::DoFHandler<dim>::active_cell_iterator>
      patch_to_global_map;
  for (auto cell : dof_handler.active_cell_iterators())
    patch_to_global_map[cell] = cell;

  dealii::AffineConstraints<double> constraints;
  DiagonalTestMeshEvaluator<dim> evaluator(cuda_handle, dof_handler,
                                           constraints);
  double *eigenvalues_dev;
  double *eigenvectors_dev;
  double *diag_elements_dev;
  std::vector<dealii::types::global_dof_index> dof_indices_map;
  std::tie(eigenvalues_dev, eigenvectors_dev, diag_elements_dev,
           dof_indices_map) =
      amge.compute_local_eigenvectors(n_eigenvectors, triangulation,
                                      patch_to_global_map, evaluator);

  unsigned int const n_dofs = dof_handler.n_dofs();
  std::vector<dealii::types::global_dof_index> ref_dof_indices_map(n_dofs);
  std::iota(ref_dof_indices_map.begin(), ref_dof_indices_map.end(), 0);
  // We cannot use BOOST_TEST because it uses variadic template and there is
  // bug in CUDA 7.0 and CUDA 8.0 with variadic templates
  // See http://www.boost.org/doc/libs/1_66_0/boost/config/compiler/nvcc.hpp
  for (unsigned int i = 0; i < n_dofs; ++i)
    BOOST_CHECK_EQUAL(dof_indices_map[i], ref_dof_indices_map[i]);

  unsigned int const eigenvector_size = 30;
  std::vector<double> ref_eigenvalues(n_eigenvectors);
  std::vector<dealii::Vector<double>> ref_eigenvectors(
      n_eigenvectors, dealii::Vector<double>(eigenvector_size));
  for (unsigned int i = 0; i < n_eigenvectors; ++i)
  {
    ref_eigenvalues[i] = static_cast<double>(i + 1);
    ref_eigenvectors[i][i] = 1.;
  }

  cudaError_t cuda_error_code;
  for (unsigned int i = 0; i < n_eigenvectors; ++i)
  {
    std::vector<double> eigenvalues(n_eigenvectors);
    cuda_error_code =
        cudaMemcpy(&eigenvalues[0], eigenvalues_dev,
                   n_eigenvectors * sizeof(double), cudaMemcpyDeviceToHost);
    mfmg::ASSERT_CUDA(cuda_error_code);
    BOOST_CHECK_CLOSE(eigenvalues[i], ref_eigenvalues[i], 1e-12);

    std::vector<double> eigenvectors(n_eigenvectors * eigenvector_size);
    cuda_error_code =
        cudaMemcpy(&eigenvectors[0], eigenvectors_dev,
                   n_eigenvectors * eigenvector_size * sizeof(double),
                   cudaMemcpyDeviceToHost);
    mfmg::ASSERT_CUDA(cuda_error_code);
    for (unsigned int j = 0; j < eigenvector_size; ++j)
      BOOST_CHECK_CLOSE(std::abs(eigenvectors[i * eigenvector_size + j]),
                        ref_eigenvectors[i][j], 1e-12);
  }

  // Free memory allocated on device
  mfmg::cuda_free(eigenvalues_dev);
  mfmg::cuda_free(eigenvectors_dev);
  mfmg::cuda_free(diag_elements_dev);
}
