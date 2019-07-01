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

#ifndef MFMG_CUDA_MATRIX_FREE_SMOOTHER
#define MFMG_CUDA_MATRIX_FREE_SMOOTHER

#include <mfmg/common/operator.hpp>
#include <mfmg/common/smoother.hpp>

#ifdef MFMG_WITH_CUDA

#include <mfmg/cuda/sparse_matrix_device.cuh>

namespace mfmg
{
template <int dim, typename VectorType>
class CudaMatrixFreeSmoother final : public Smoother<VectorType>
{
public:
  using value_type = typename VectorType::value_type;
  using vector_type = VectorType;

  CudaMatrixFreeSmoother(
      std::shared_ptr<Operator<vector_type> const> op,
      std::shared_ptr<boost::property_tree::ptree const> params);

  virtual void apply(vector_type const &x, vector_type &y) const override;

private:
  SparseMatrixDevice<value_type> _smoother;
};
} // namespace mfmg

#endif

#endif
