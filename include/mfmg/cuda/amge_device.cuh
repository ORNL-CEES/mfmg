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

#ifndef AMGE_DEVICE_CUH
#define AMGE_DEVICE_CUH

#include <mfmg/common/amge.hpp>
#include <mfmg/common/exceptions.hpp>
#include <mfmg/cuda/cuda_handle.cuh>
#include <mfmg/cuda/cuda_matrix_free_mesh_evaluator.cuh>
#include <mfmg/cuda/cuda_mesh_evaluator.cuh>
#include <mfmg/cuda/sparse_matrix_device.cuh>

#include <deal.II/lac/cuda_vector.h>

namespace mfmg
{
template <int dim, typename MeshEvaluator, typename VectorType>
class AMGe_device : public AMGe<dim, VectorType>
{
public:
  using ScalarType = typename VectorType::value_type;

  AMGe_device(MPI_Comm comm, dealii::DoFHandler<dim> const &dof_handler,
              CudaHandle const &cuda_handle,
              boost::property_tree::ptree const &eigensolver_params =
                  boost::property_tree::ptree());

  /**
   * Compute the eigenvalues, the eigenvectors, the local diagonal elements, the
   * map between local and global dof indices. This functions takes as inputs:
   *  - the number of eigenvalues to compute
   *  - the triangulation associated to the agglomerate
   *  - the map between the local cells and the global cells
   *  - an object that can evaluate the user's global operator and local
   *  operator
   */
  // The function cannot be const because we use the handles
  template <typename TriangulationType>
  std::tuple<ScalarType *, ScalarType *, ScalarType *,
             std::vector<dealii::types::global_dof_index>>
  compute_local_eigenvectors(
      unsigned int n_eigenvectors, double const tolerance,
      TriangulationType const &agglomerate_triangulation,
      std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
               typename dealii::DoFHandler<dim>::active_cell_iterator> const
          &patch_to_global_map,
      MeshEvaluator const &evaluator,
      typename std::enable_if_t<is_matrix_free<MeshEvaluator>::value &&
                                    std::is_class<TriangulationType>::value,
                                int> = 0);

  template <typename TriangulationType>
  std::tuple<ScalarType *, ScalarType *, ScalarType *,
             std::vector<dealii::types::global_dof_index>>
  compute_local_eigenvectors(
      unsigned int n_eigenvectors, double const tolerance,
      TriangulationType const &agglomerate_triangulation,
      std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
               typename dealii::DoFHandler<dim>::active_cell_iterator> const
          &patch_to_global_map,
      MeshEvaluator const &evaluator,

      typename std::enable_if_t<!is_matrix_free<MeshEvaluator>::value &&
                                    std::is_class<TriangulationType>::value,
                                int> = 0);

  /**
   * Compute the restriction sparse matrix. The rows of the matrix are the
   * weighted eigenvectors. \p dof_indices_maps are used to map the indices in
   * \p eigenvectors to the global dof indices.
   */
  SparseMatrixDevice<typename VectorType::value_type>
  compute_restriction_sparse_matrix(
      std::vector<dealii::Vector<typename VectorType::value_type>> const
          &eigenvectors,
      std::vector<std::vector<typename VectorType::value_type>> const
          &diag_elements,
      dealii::LinearAlgebra::distributed::Vector<
          typename VectorType::value_type> const &locally_relevant_global_diag,
      std::vector<std::vector<dealii::types::global_dof_index>> const
          &dof_indices_maps,
      std::vector<unsigned int> const &n_local_eigenvectors,
      cusparseHandle_t cusparse_handle);

  /**
   *  Build the agglomerates and their associated triangulations.
   */
  SparseMatrixDevice<typename VectorType::value_type>
  setup_restrictor(boost::property_tree::ptree const &agglomerate_dim,
                   unsigned int const n_eigenvectors, double const tolerance,
                   MeshEvaluator const &evaluator);

private:
  CudaHandle const &_cuda_handle;
  boost::property_tree::ptree _eigensolver_params;
};
} // namespace mfmg

#endif
