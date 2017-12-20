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

#ifndef AMGE_HPP
#define AMGE_HPP

#include <deal.II/base/mpi.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include <array>
#include <map>
#include <string>
#include <tuple>

namespace mfmg
{
template <int dim, typename NumberType>
class AMGe
{
public:
  AMGe(MPI_Comm comm, dealii::DoFHandler<dim> const &dof_handler);

  /**
   * Flag cells to create agglomerates. The desired size of the agglomerates is
   * given by \p agglomerate_dim. This functions returns the number of
   * agglomerates that have been created.
   */
  unsigned int build_agglomerates(
      std::array<unsigned int, dim> const &agglomerate_dim) const;

  /**
   * Create a Triangulation \p agglomerate_triangulation associated with an
   * agglomerate of a given \p agglomerate_id and a map that matches cells in
   * the local triangulation with cells in the global triangulation.
   */
  std::tuple<dealii::Triangulation<dim>,
             std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
                      typename dealii::DoFHandler<dim>::active_cell_iterator>>
  build_agglomerate_triangulation(unsigned int agglomerate_id) const;

  /**
   * Compute the eigenvalues and the eigenvectors. This functions takes as
   * inputs:
   *  - the number of eigenvalues to compute
   *  - the tolerance (relative accuracy of the Ritz value)
   *  - the map between the local cells and the global cells
   *  - a function that computes the local DoFHandler, the local system sparse
   *    matrix with its sparsity pattern, the local mass matrix with its
   *    sparsity pattern, and local sparse matrix.
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
      std::function<void(dealii::DoFHandler<dim> &dof_handler,
                         dealii::SparsityPattern &system_sparsity_pattern,
                         dealii::SparseMatrix<NumberType> &,
                         dealii::SparsityPattern &mass_sparsity_pattern,
                         dealii::SparseMatrix<NumberType> &,
                         dealii::ConstraintMatrix &)> const &evaluate) const;

  /**
   * Output the mesh and the agglomerate ids.
   */
  void output(std::string const &filename) const;

  /**
   *  Build the agglomerates and their associated triangulations.
   */
  void setup(std::array<unsigned int, dim> const &agglomerate_dim);

private:
  /**
   * This data structure is empty but it is necessary to use WorkStream.
   */
  struct ScratchData
  {
    // nothing
  };

  /**
   * This data structure is empty but it is necessary to use WorkStream.
   */
  struct CopyData
  {
    // nothing
  };

  /**
   * This function encapsulates the different functions that work on an
   * independent set of data.
   */
  void local_worker(std::vector<unsigned int>::iterator const &agg_id,
                    ScratchData &scratch_data, CopyData &copy_data);

  /**
   * This function does nothing but is necessary to use WorkStream.
   */
  void copy_local_to_global(CopyData const &copy_data);

  /**
   * Compute the map between the dof indices of the local DoFHandler and the dof
   * indices of the global DoFHandler.
   */
  std::vector<dealii::types::global_dof_index> compute_dof_index_map(
      std::map<typename dealii::Triangulation<dim>::active_cell_iterator,
               typename dealii::DoFHandler<dim>::active_cell_iterator> const
          &patch_to_global_map,
      dealii::DoFHandler<dim> const &agglomerate_dof_handler) const;

  MPI_Comm _comm;
  dealii::DoFHandler<dim> const &_dof_handler;
};
}

#endif
