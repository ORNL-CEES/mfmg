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

#ifndef UTILS_H
#define UTILS_H

#include <mfmg/common/exceptions.hpp>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <Teuchos_ParameterList.hpp>

#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

namespace mfmg
{
void ptree2plist(boost::property_tree::ptree const &ptree,
                 Teuchos::ParameterList &plist);

template <typename T>
std::vector<unsigned int> sort_permutation(std::vector<T> const &vec_1,
                                           std::vector<T> const &vec_2)
{
  ASSERT(vec_1.size() == vec_2.size(), "The vectors need to have the same size "
                                       "but vector 1 has a size " +
                                           std::to_string(vec_1.size()) +
                                           " and vector 2 has a size " +
                                           std::to_string(vec_2.size()));

  std::vector<unsigned int> permutation(vec_1.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [&](unsigned int i, unsigned int j) {
              if (vec_1[i] != vec_1[j])
                return (vec_1[i] < vec_1[j]);
              else
                return (vec_2[i] < vec_2[j]);
            });

  return permutation;
}

template <typename T>
void apply_permutation_in_place(std::vector<unsigned int> const &permutation,
                                std::vector<T> &vec)
{
  unsigned int const n_elem = vec.size();
  std::vector<bool> done(n_elem, false);
  for (unsigned int i = 0; i < n_elem; ++i)
  {
    if (done[i] == false)
    {
      done[i] = true;
      unsigned int prev_j = i;
      unsigned int j = permutation[i];
      while (i != j)
      {
        std::swap(vec[prev_j], vec[j]);
        done[j] = true;
        prev_j = j;
        j = permutation[j];
      }
    }
  }
}

template <typename ScalarType>
void check_restriction_matrix(
    MPI_Comm comm, std::vector<dealii::Vector<ScalarType>> const &eigenvectors,
    std::vector<std::vector<dealii::types::global_dof_index>> const
        &dof_indices_maps,
    dealii::LinearAlgebra::distributed::Vector<ScalarType> const
        &locally_relevant_global_diag,
    std::vector<std::vector<ScalarType>> const &diag_elements,
    std::vector<unsigned int> const &n_local_eigenvectors)
{
#if MFMG_DEBUG
  // Check that the locally_relevant_global_diag is the sum of the agglomerates
  // diagonal
  // TODO do not ask user for the locally_relevant_global_diag
  dealii::LinearAlgebra::distributed::Vector<ScalarType> new_global_diag(
      locally_relevant_global_diag.get_partitioner());
  unsigned int const n_agglomerates = n_local_eigenvectors.size();
  for (unsigned int i = 0; i < n_agglomerates; ++i)
  {
    // Get the size of the eigenvectors in agglomerate i
    unsigned int offset = std::accumulate(n_local_eigenvectors.begin(),
                                          n_local_eigenvectors.begin() + i, 0);
    unsigned int const n_elem = eigenvectors[offset].size();

    for (unsigned int j = 0; j < n_elem; ++j)
    {
      dealii::types::global_dof_index const global_pos = dof_indices_maps[i][j];
      new_global_diag[global_pos] += diag_elements[i][j];
    }
  }
  new_global_diag.compress(dealii::VectorOperation::add);
  new_global_diag -= locally_relevant_global_diag;
  ASSERT((new_global_diag.linfty_norm() /
          locally_relevant_global_diag.linfty_norm()) < 1e-14,
         "Sum of agglomerate diagonals is not equal to the global diagonal");

  // Check that the sum of the weight matrices is the identity
  auto locally_owned_dofs =
      locally_relevant_global_diag.locally_owned_elements();
  dealii::TrilinosWrappers::SparsityPattern sp(locally_owned_dofs,
                                               locally_owned_dofs, comm);
  for (auto local_index : locally_owned_dofs)
    sp.add(local_index, local_index);
  sp.compress();

  dealii::TrilinosWrappers::SparseMatrix weight_matrix(sp);
  unsigned int pos = 0;
  for (unsigned int i = 0; i < n_agglomerates; ++i)
  {
    unsigned int const n_elem = eigenvectors[pos].size();
    for (unsigned int j = 0; j < n_elem; ++j)
    {
      dealii::types::global_dof_index const global_pos = dof_indices_maps[i][j];
      double const value =
          diag_elements[i][j] / locally_relevant_global_diag[global_pos];
      weight_matrix.add(global_pos, global_pos, value);
    }
    pos += n_local_eigenvectors[i];
  }

  // Compress the matrix
  weight_matrix.compress(dealii::VectorOperation::add);

  for (auto index : locally_owned_dofs)
    ASSERT(std::abs(weight_matrix.diag_element(index) - 1.0) < 1e-14,
           "Sum of local weight matrices is not the identity");
#endif
}
} // namespace mfmg

#endif
