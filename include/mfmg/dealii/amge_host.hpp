/*************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                           *
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

#include <mfmg/common/amge.hpp>

namespace mfmg
{
template <int dim, typename MeshEvaluator, typename VectorType>
class AMGe_host : public AMGe<dim, VectorType>
{
public:
  using ScalarType = typename VectorType::value_type;

  AMGe_host(MPI_Comm comm, dealii::DoFHandler<dim> const &dof_handler,
            boost::property_tree::ptree const &eigensolver_params =
                boost::property_tree::ptree());

  /**
   * Compute the eigenvalues and the eigenvectors. This functions takes as
   * inputs:
   *  - the number of eigenvalues to compute
   *  - the tolerance (relative accuracy of the Ritz value)
   *  - the triangulation of the agglomerate
   *  - the map between the local cells and the global cells
   *  - an object that can evaluates the local DoFHandler, the local
   *    AffineConstraints<double>, and the local system sparse matrix with its
   *    sparsity pattern.
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
   *  Build the agglomerates and their associated triangulations.
   */
  void setup_restrictor(
      boost::property_tree::ptree const &params,
      unsigned int const n_eigenvectors, double const tolerance,
      MeshEvaluator const &evaluator,
      dealii::LinearAlgebra::distributed::Vector<
          typename VectorType::value_type> const &locally_relevant_global_diag,
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

  boost::property_tree::ptree _eigensolver_params;
};
} // namespace mfmg

#endif
