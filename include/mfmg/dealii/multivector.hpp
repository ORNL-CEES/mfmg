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

#ifndef MFMG_MULTIVECTOR_HPP
#define MFMG_MULTIVECTOR_HPP

#include <mfmg/common/exceptions.hpp>

#include <algorithm>
#include <memory>
#include <vector>

namespace mfmg
{
template <typename VectorType>
class MultiVector
{
public:
  MultiVector(int n_vectors, int vector_size = 0)
  {
    _vectors.resize(n_vectors);
    std::for_each(_vectors.begin(), _vectors.end(),
                  [](auto &v) { v = std::make_shared<VectorType>(); });
    if (vector_size)
      std::for_each(_vectors.begin(), _vectors.end(),
                    [vector_size](auto &v) { v->reinit(vector_size); });
  }
  int size() const { return _vectors.empty() ? 0 : _vectors[0]->size(); }
  int n_vectors() const { return _vectors.size(); }
  std::shared_ptr<VectorType> &operator[](int index) { return _vectors[index]; }
  std::shared_ptr<VectorType> const &operator[](int index) const
  {
    return _vectors[index];
  }

private:
  std::vector<std::shared_ptr<VectorType>> _vectors;
};
} // namespace mfmg

#endif
