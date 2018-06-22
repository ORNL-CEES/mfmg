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

#ifndef AMGE_HOST_HPP
#define AMGE_HOST_HPP

#include <mfmg/amge.hpp>

namespace mfmg
{
template <int dim, typename MeshEvaluator, typename VectorType>
class AMGe_host : public AMGe<dim, VectorType>
{
public:
  using ScalarType = typename VectorType::value_type;

  AMGe_host(MPI_Comm comm, dealii::DoFHandler<dim> const &dof_handler,
            std::string const eigensolver_type = "arpack");

  /**
   * Compute the eigenvalues and the eigenvectors. This functions takes as
   * inputs:
   *  - the number of eigenvalues to compute
   *  - the tolerance (relative accuracy of the Ritz value)
   *  - the triangulation of the agglomerate
   *  - the map between the local cells and the global cells
   *  - an object that can evaluates the local DoFHandler, the local
   *    ConstraintMatrix, and the local system sparse matrix with its sparsity
   *    pattern.
   *
   * The function returns the complex eigenvalues, the associated eigenvectors,
   * the diagonal elements of the local system matrix, and a vector that maps
   * the dof indices from the local problem to the global problem.
   */
  std::tuple<std::vector<std::complex<double>>,
             std::vector<dealii::Vector<double>>, std::vector<ScalarType>,
             std::vector<dealii::types::global_dof_index>>
  compute_local_eigenvectors(
      unsigned int n_eigenvectors, double tolerance,
      dealii::Triangulation<dim> const &agglomerate_triangulation,
      std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
               typename dealii::DoFHandler<dim>::active_cell_iterator> const
          &patch_to_global_map,
      MeshEvaluator const &evaluator) const;

  /**
   * Compute the restriction sparse matrix. The rows of the matrix are
   * weighted eigenvectors. \p dof_indices_maps are used to map the indices in
   * \p eigenvectors to the global dof indices.
   */
  // dealii::TrilinosWrappers::SparseMatrix has a private copy constructor and
  // no move constructor. Thus, we pass the output by reference instead of
  // returning it.
  void compute_restriction_sparse_matrix(
      std::vector<dealii::Vector<double>> const &eigenvectors,
      std::vector<std::vector<ScalarType>> const &diag_elements,
      std::vector<std::vector<dealii::types::global_dof_index>> const
          &dof_indices_map,
      std::vector<unsigned int> const &n_local_eigenvectors,
      dealii::TrilinosWrappers::SparseMatrix const &system_sparse_matrix,
      dealii::TrilinosWrappers::SparseMatrix &restriction_sparse_matrix) const;

  /**
   *  Build the agglomerates and their associated triangulations.
   */
  void setup_restrictor(
      std::array<unsigned int, dim> const &agglomerate_dim,
      unsigned int const n_eigenvectors, double const tolerance,
      MeshEvaluator const &evaluator,
      std::shared_ptr<typename MeshEvaluator::global_operator_type const>
          global_operator,
      dealii::TrilinosWrappers::SparseMatrix &restriction_sparse_matrix);

private:
  /**
   * This data structure is empty but it is necessary to use WorkStream.
   */
  struct ScratchData
  {
    // nothing
  };

  /**
   * Structure which encapsulates the data that needs to be copied add the end
   * of Worstream.
   */
  struct CopyData
  {
    std::vector<dealii::Vector<double>> local_eigenvectors;
    std::vector<ScalarType> diag_elements;
    std::vector<dealii::types::global_dof_index> local_dof_indices_map;
  };

  /**
   * This function encapsulates the different functions that work on an
   * independent set of data.
   */
  void local_worker(unsigned int const n_eigenvectors, double const tolerance,
                    MeshEvaluator const &evalute,
                    std::vector<unsigned int>::iterator const &agg_id,
                    ScratchData &scratch_data, CopyData &copy_data);

  /**
   * This function copies quantities computed in local worker to output
   * variables.
   */
  void
  copy_local_to_global(CopyData const &copy_data,
                       std::vector<dealii::Vector<double>> &eigenvectors,
                       std::vector<std::vector<ScalarType>> &diag_elements,
                       std::vector<std::vector<dealii::types::global_dof_index>>
                           &dof_indices_maps,
                       std::vector<unsigned int> &n_local_eigenvectors);

  std::string const _eigensolver_type;
};
} // namespace mfmg

#endif
