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

#ifndef UTILS_H
#define UTILS_H

#include <mfmg/common/exceptions.hpp>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <Teuchos_ParameterList.hpp>

#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <vector>

namespace mfmg
{

void matrix_market_output_file(
    const std::string &filename,
    const dealii::TrilinosWrappers::SparseMatrix &matrix);

void matrix_market_output_file(
    const std::string &filename,
    const dealii::TrilinosWrappers::MPI::Vector &vector);

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
} // namespace mfmg

#endif
