/*************************************************************************
 * Copyright (c) 2017 - 2018 by the mfmg authors                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#ifndef AMGE_DEVICE_CUH
#define AMGE_DEVICE_CUH

#include <mfmg/amge.hpp>
#include <mfmg/exceptions.hpp>
#include <mfmg/sparse_matrix_device.cuh>

#include <deal.II/lac/cuda_vector.h>

#include <cusolverDn.h>
#include <cusparse.h>

namespace mfmg
{
template <int dim, typename VectorType>
class AMGe_device : public AMGe<dim, VectorType>
{
public:
  using ScalarType = typename VectorType::value_type;

  AMGe_device(MPI_Comm comm, dealii::DoFHandler<dim> const &dof_handler,
              cusolverDnHandle_t cusolver_dn_handle,
              cusparseHandle_t cusparse_handle);

  /**
   * Compute the eigenvalues and the eigenvectors. This functions takes as
   * inputs:
   *  - the number of eigenvalues to compute
   *  - the triangulation associated to the agglomerate
   *  - the map between the local cells and the global cells
   *  - a function that evaluates the local DoFHandler, the local
   *  ConstraintMatrix, the system matrix (on the device), and the mass matrix
   * (on the device)
   *
   * The function returns a pointer to a 1D array (on the device) with the
   * eigenvalues, a pointer to a 1D array (on the device) containing the
   * associated eigenvectors, and a vector that maps the dof indices from the
   * local problem to the global problem.
   */
  // The function cannot be const because we use the handles
  std::tuple<ScalarType *, ScalarType *,
             std::vector<dealii::types::global_dof_index>>
  compute_local_eigenvectors(
      unsigned int n_eigenvalues,
      dealii::Triangulation<dim> const &agglomerate_triangulation,
      std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
               typename dealii::DoFHandler<dim>::active_cell_iterator> const
          &patch_to_global_map,
      std::function<void(
          dealii::DoFHandler<dim> &, dealii::ConstraintMatrix &,
          std::shared_ptr<SparseMatrixDevice<ScalarType>> &,
          std::shared_ptr<SparseMatrixDevice<ScalarType>> &)> const &evaluate);

  /*
   * Compute the restriction sparse matrix. The rows of the matrix are the
   * computed eigenvectors. \p dof_indices_maps are used to map the indices in
   * \p eigenvectors_dev to the global dof indices.
   */
  SparseMatrixDevice<ScalarType> compute_restriction_sparse_matrix(
      ScalarType *eigenvectors_dev,
      std::vector<std::vector<dealii::types::global_dof_index>> const
          &dof_indices_map);

private:
  cusolverDnHandle_t _cusolver_dn_handle;
  cusparseHandle_t _cusparse_handle;
};
}

#endif
