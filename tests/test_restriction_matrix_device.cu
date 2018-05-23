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
#include <mfmg/dealii_adapters_device.cuh>
#include <mfmg/sparse_matrix_device.cuh>
#include <mfmg/utils.cuh>

#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/la_parallel_vector.h>

template <int dim, typename VectorType>
class DummyMeshEvaluator
    : public mfmg::DealIIMeshEvaluatorDevice<dim, VectorType>
{
public:
  using value_type = typename VectorType::value_type;

  DummyMeshEvaluator(mfmg::CudaHandle &cuda_handle)
      : mfmg::DealIIMeshEvaluatorDevice<dim, VectorType>(cuda_handle)
  {
  }

  virtual dealii::LinearAlgebra::distributed::Vector<value_type>
  get_locally_relevant_diag() const override final
  {
    return dealii::LinearAlgebra::distributed::Vector<value_type>();
  }

protected:
  virtual void
  evaluate_local(dealii::DoFHandler<dim> &, dealii::ConstraintMatrix &,
                 std::shared_ptr<mfmg::SparseMatrixDevice<value_type>> &)
      const override final
  {
  }

  virtual void
  evaluate_global(dealii::DoFHandler<dim> &, dealii::ConstraintMatrix &,
                  std::shared_ptr<mfmg::SparseMatrixDevice<value_type>> &)
      const override final
  {
  }
};

BOOST_AUTO_TEST_CASE(restriction_matrix)
{
  unsigned int constexpr dim = 2;
  using Vector = dealii::LinearAlgebra::distributed::Vector<double>;
  using value_type = typename Vector::value_type;

  mfmg::CudaHandle cuda_handle;

  MPI_Comm comm = MPI_COMM_WORLD;
  dealii::parallel::distributed::Triangulation<dim> triangulation(comm);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);
  dealii::FE_Q<dim> fe(1);
  dealii::DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  DummyMeshEvaluator<dim, Vector> evaluator(cuda_handle);
  mfmg::AMGe_device<dim, mfmg::DealIIMeshEvaluatorDevice<dim, Vector>, Vector>
      amge(comm, dof_handler, cuda_handle);

  auto const locally_owned_dofs = dof_handler.locally_owned_dofs();
  unsigned int const n_local_rows = locally_owned_dofs.n_elements();

  // Fill the eigenvectors
  unsigned int const eigenvectors_size = 3;
  std::vector<dealii::Vector<double>> eigenvectors(
      n_local_rows, dealii::Vector<double>(eigenvectors_size));
  for (unsigned int i = 0; i < n_local_rows; ++i)
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
      eigenvectors[i][j] =
          n_local_rows * eigenvectors_size + i * eigenvectors_size + j;

  // Fill dof_indices_maps
  std::vector<std::vector<dealii::types::global_dof_index>> dof_indices_maps(
      n_local_rows,
      std::vector<dealii::types::global_dof_index>(eigenvectors_size));
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, dof_handler.n_dofs() - 1);
  for (unsigned int i = 0; i < n_local_rows; ++i)
  {
    // We don't want dof_indices to have repeated values in a row
    std::set<int> dofs_set;
    unsigned int j = 0;
    while (dofs_set.size() < eigenvectors_size)
    {
      int dof_index = distribution(generator);
      if ((dofs_set.count(dof_index) == 0) &&
          (locally_owned_dofs.is_element(dof_index)))
      {
        dof_indices_maps[i][j] = dof_index;
        dofs_set.insert(dof_index);
        ++j;
      }
    }
  }

  for (unsigned int i = 0; i < n_local_rows; ++i)
    std::sort(dof_indices_maps[i].begin(), dof_indices_maps[i].end());

  // Fill diag_elements
  std::vector<std::vector<double>> diag_elements(
      n_local_rows, std::vector<double>(eigenvectors_size));
  std::map<unsigned int, double> count_elem;
  for (unsigned int i = 0; i < n_local_rows; ++i)
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
      count_elem[dof_indices_maps[i][j]] += 1.0;
  for (unsigned int i = 0; i < n_local_rows; ++i)
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
      diag_elements[i][j] = 1. / count_elem[dof_indices_maps[i][j]];

  // Fill n_local_eigenvectors
  std::vector<unsigned int> n_local_eigenvectors(n_local_rows, 1);

  // Fill system_sparse_matrix
  dealii::TrilinosWrappers::SparseMatrix system_sparse_matrix(
      locally_owned_dofs, comm);
  for (auto const index : locally_owned_dofs)
    system_sparse_matrix.set(index, index, 1.0);
  system_sparse_matrix.compress(dealii::VectorOperation::insert);

  // Fill locally_relevant_global_diag
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                  locally_relevant_dofs);
  Vector locally_relevant_global_diag(locally_owned_dofs, locally_relevant_dofs,
                                      MPI_COMM_WORLD);
  for (auto &val : locally_relevant_global_diag)
    val = 1.;
  locally_relevant_global_diag.compress(dealii::VectorOperation::insert);

  mfmg::SparseMatrixDevice<value_type> restriction_matrix_dev =
      amge.compute_restriction_sparse_matrix(
          eigenvectors, diag_elements, locally_relevant_global_diag,
          dof_indices_maps, n_local_eigenvectors, cuda_handle.cusparse_handle);

  // Move the values to the host
  std::vector<value_type> restriction_matrix_host(
      restriction_matrix_dev.local_nnz());
  mfmg::cuda_mem_copy_to_host(restriction_matrix_dev.val_dev,
                              restriction_matrix_host);

  std::vector<int> column_index_host(restriction_matrix_dev.local_nnz());
  mfmg::cuda_mem_copy_to_host(restriction_matrix_dev.column_index_dev,
                              column_index_host);

  // Check that the matrix was built correctly
  unsigned int pos = 0;
  for (unsigned int i = 0; i < n_local_rows; ++i)
  {
    for (unsigned int j = 0; j < eigenvectors_size; ++j)
    {
      BOOST_CHECK_EQUAL(restriction_matrix_host[pos],
                        diag_elements[i][j] * eigenvectors[i][j]);
      ++pos;
    }
  }
}
