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

#include <mfmg/common/lanczos.templates.hpp>

namespace mfmg
{
namespace internal
{
#ifdef __CUDACC__
template <>
void details_set_initial_guess(
    dealii::LinearAlgebra::distributed::Vector<
        double, dealii::MemorySpace::CUDA> &initial_guess,
    int seed)
{
  // Modify initial guess with a random noise by multiplying each entry of the
  // vector with a random value from a uniform distribution. This specific
  // procedure guarantees that zero entries of the vector stay zero, which is
  // important for situations where they are associated with constrained dofs in
  // Deal.II
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0, 1);
  std::vector<double> initial_guess_host(initial_guess.local_size());
  cuda_mem_copy_to_host(initial_guess.get_values(), initial_guess_host);
  std::transform(initial_guess_host.begin(), initial_guess_host.end(),
                 initial_guess_host.begin(),
                 [&](auto &v) { return (1. + dist(gen)) * v; });
  cuda_mem_copy_to_dev(initial_guess_host, initial_guess.get_values());
}
#endif
} // namespace internal
} // namespace mfmg
