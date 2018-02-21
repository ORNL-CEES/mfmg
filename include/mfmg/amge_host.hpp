/*************************************************************************
 * Copyright (c) 2017 by the mfmg authors                                *
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
template <int dim, typename VectorType>
class AMGe_host : public AMGe<dim, VectorType>
{
public:
  using ScalarType = typename VectorType::value_type;

  AMGe_host(MPI_Comm comm, dealii::DoFHandler<dim> const &dof_handler);

  AMGe_host(AMGe_host<dim, VectorType> const &other);

  /**
   * Compute the eigenvalues and the eigenvectors. This functions takes as
   * inputs:
   *  - the number of eigenvalues to compute
   *  - the tolerance (relative accuracy of the Ritz value)
   *  - the triangulation of the agglomerate
   *  - the map between the local cells and the global cells
   *  - a function that evaluates the local DoFHandler, the local
   *    ConstraintMatrix, the local system sparse matrix with its sparsity
   *    pattern, and the local mass matrix with its sparsity pattern.
   *
   * The function returns the complex eigenvalues, the associated eigenvectors,
   * and a vector that maps the dof indices from the local problem to the global
   * problem.
   */
  std::tuple<std::vector<std::complex<double>>,
             std::vector<dealii::Vector<double>>,
             std::vector<dealii::types::global_dof_index>>
  compute_local_eigenvectors(
      unsigned int n_eigenvalues, double tolerance,
      dealii::Triangulation<dim> const &agglomerate_triangulation,
      std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
               typename dealii::DoFHandler<dim>::active_cell_iterator> const
          &patch_to_global_map,
      std::function<
          void(dealii::DoFHandler<dim> &dof_handler, dealii::ConstraintMatrix &,
               dealii::SparsityPattern &system_sparsity_pattern,
               dealii::SparseMatrix<ScalarType> &,
               dealii::SparsityPattern &mass_sparsity_pattern,
               dealii::SparseMatrix<ScalarType> &)> const &evaluate) const;

  /**
   * Compute the restriction sparse matrix. The rows of the matrix are the
   * computed eigenvectors. \p dof_indices_maps are used to map the indices in
   * \p eigenvectors to the global dof indices.
   */
  // dealii::TrilinosWrappers::SparseMatrix has a private copy constructor and
  // no move constructor. Thus, we pass the output by reference instead of
  // returning it.
  void compute_restriction_sparse_matrix(
      std::vector<dealii::Vector<double>> const &eigenvectors,
      std::vector<std::vector<dealii::types::global_dof_index>> const
          &dof_indices_map,
      dealii::TrilinosWrappers::SparseMatrix &restriction_sparse_matrix) const;

  /**
   *  Build the agglomerates and their associated triangulations.
   */
  void
  setup(std::array<unsigned int, dim> const &agglomerate_dim,
        unsigned int const n_eigenvalues, double const tolerance,
        std::function<void(dealii::DoFHandler<dim> &dof_handler,
                           dealii::ConstraintMatrix &,
                           dealii::SparsityPattern &system_sparsity_pattern,
                           dealii::SparseMatrix<ScalarType> &,
                           dealii::SparsityPattern &mass_sparsity_pattern,
                           dealii::SparseMatrix<ScalarType> &)> const &evaluate,
        dealii::TrilinosWrappers::SparseMatrix const &system_sparse_matrix);

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
    std::vector<dealii::types::global_dof_index> local_dof_indices_map;
  };

  /**
   * This function encapsulates the different functions that work on an
   * independent set of data.
   */
  void local_worker(
      unsigned int const n_eigenvalues, double const tolerance,
      std::function<void(dealii::DoFHandler<dim> &dof_handler,
                         dealii::ConstraintMatrix &,
                         dealii::SparsityPattern &system_sparsity_pattern,
                         dealii::SparseMatrix<ScalarType> &,
                         dealii::SparsityPattern &mass_sparsity_pattern,
                         dealii::SparseMatrix<ScalarType> &)> const &evaluate,
      std::vector<unsigned int>::iterator const &agg_id,
      ScratchData &scratch_data, CopyData &copy_data);

  /**
   * This function does nothing but is necessary to use WorkStream.
   */
  void
  copy_local_to_global(CopyData const &copy_data,
                       std::vector<dealii::Vector<double>> &eigenvectors,
                       std::vector<std::vector<dealii::types::global_dof_index>>
                           &dof_indices_maps);

  /**
   * Build the sparsity pattern of the restriction matrix, i.e., the
   * Epetra_FECrsGraph in Trilinos nomenclature.
   */
  dealii::TrilinosWrappers::SparsityPattern
  compute_restriction_sparsity_pattern(
      std::vector<dealii::Vector<double>> const &eigenvectors,
      std::vector<std::vector<dealii::types::global_dof_index>> const
          &dof_indices_maps) const;

  dealii::TrilinosWrappers::SparseMatrix const *_system_matrix_ptr;
  dealii::TrilinosWrappers::SparseMatrix _restriction_sparse_matrix;
  dealii::TrilinosWrappers::SparseMatrix _coarse_sparse_matrix;
};
}

#endif
