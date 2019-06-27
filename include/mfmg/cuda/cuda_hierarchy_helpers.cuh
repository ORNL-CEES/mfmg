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

#ifndef MFMG_CUDA_HIERARCHY_HELPERS_CUH
#define MFMG_CUDA_HIERARCHY_HELPERS_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/common/hierarchy_helpers.hpp>
#include <mfmg/cuda/cuda_handle.cuh>

namespace mfmg
{
template <int dim, typename VectorType>
class CudaHierarchyHelpers : public HierarchyHelpers<VectorType>
{
public:
  using vector_type = VectorType;

  CudaHierarchyHelpers(CudaHandle const &cuda_handle);

  virtual ~CudaHierarchyHelpers() override = default;

  std::shared_ptr<Operator<vector_type>>
  get_global_operator(std::shared_ptr<MeshEvaluator> mesh_evaluator) override;

  std::shared_ptr<Operator<vector_type>> build_restrictor(
      MPI_Comm comm, std::shared_ptr<MeshEvaluator> mesh_evaluator,
      std::shared_ptr<boost::property_tree::ptree const> params) override;

  std::shared_ptr<Smoother<vector_type>> build_smoother(
      std::shared_ptr<Operator<vector_type> const> op,
      std::shared_ptr<boost::property_tree::ptree const> params) override;

  std::shared_ptr<Solver<vector_type>> build_coarse_solver(
      std::shared_ptr<Operator<vector_type> const> op,
      std::shared_ptr<boost::property_tree::ptree const> params) override;

protected:
  CudaHandle const &_cuda_handle;
  std::shared_ptr<Operator<vector_type>> _operator;
};
} // namespace mfmg

#endif

#endif
