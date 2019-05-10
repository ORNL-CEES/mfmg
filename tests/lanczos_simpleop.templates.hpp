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

#ifndef MFMG_LANCZOS_SIMPLEOP_TEMPLATE_HPP
#define MFMG_LANCZOS_SIMPLEOP_TEMPLATE_HPP

#include <deal.II/lac/la_parallel_vector.h>

#include <cassert>

#include "lanczos_simpleop.hpp"

namespace mfmg
{
// This will be a diagonal matrix; specify the diag entries here
#ifdef __CUDACC__
__host__ __device__
#endif
    double
    diag_value(size_t i, size_t multiplicity)
{
  return 1 + i / multiplicity;
}

#ifdef __CUDACC__
__global__ void vmult_kernel(double *y, double const *x, unsigned int dim,
                             unsigned int multiplicity)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < dim)
  {
    y[i] = diag_value(i, multiplicity) * x[i];
  }
}
#endif

/// \brief Simple test operator: constructor
template <typename VectorType>
SimpleOperator<VectorType>::SimpleOperator(size_t dim, size_t multiplicity)
    : _dim(dim), _multiplicity(multiplicity)
{
  assert(multiplicity > 0);
}

template <typename VectorType>
std::vector<double> SimpleOperator<VectorType>::get_evals() const
{
  std::vector<double> evals(_dim);
  for (size_t i = 0; i < _dim; i++)
    evals[i] = diag_value(i, _multiplicity);

  return evals;
}

/// \brief Simple test operator: apply operator to a vector
template <typename VectorType>
void SimpleOperator<VectorType>::vmult(VectorType &y, VectorType const &x) const
{
  assert(x.size() == _dim);
  assert(y.size() == _dim);

  for (size_t i = 0; i < _dim; ++i)
    y[i] = diag_value(i, _multiplicity) * x[i];
}

#ifdef __CUDACC__
template <>
void SimpleOperator<dealii::LinearAlgebra::distributed::Vector<
    double, dealii::MemorySpace::CUDA>>::
    vmult(dealii::LinearAlgebra::distributed::Vector<
              double, dealii::MemorySpace::CUDA> &y,
          dealii::LinearAlgebra::distributed::Vector<
              double, dealii::MemorySpace::CUDA> const &x) const
{
  // CUDA does not like size_t. I think that size_t on the device is 32 bits
  // while it 64 bits on the host and thus, dim and multiplicity in the kernel
  // are garbage
  unsigned int d = _dim;
  unsigned int m = _multiplicity;
  int n_blocks = 1 + _dim / 512;
  vmult_kernel<<<n_blocks, 512>>>(y.get_values(), x.get_values(), d, m);
}
#endif

} // namespace mfmg

#endif
