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

#ifndef MFMG_CUDA_MATRIX_FREE_HIERARCHY_HELPERS_CUH
#define MFMG_CUDA_MATRIX_FREE_HIERARCHY_HELPERS_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/cuda/cuda_hierarchy_helpers.cuh>

namespace mfmg
{
template <int dim, typename VectorType>
class CudaMatrixFreeHierarchyHelpers
    : public CudaHierarchyHelpers<dim, VectorType>
{
public:
  using vector_type = VectorType;

  CudaMatrixFreeHierarchyHelpers(CudaHandle const &cuda_handle)
      : CudaHierarchyHelpers<dim, VectorType>(cuda_handle)
  {
  }
};
} // namespace mfmg

#endif

#endif
