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

#ifndef MFMG_CUDA_SOLVER_CUH
#define MFMG_CUDA_SOLVER_CUH

#include <mfmg/common/solver.hpp>

#if defined(MFMG_WITH_CUDA) && defined(__CUDACC__)

#include <mfmg/cuda/cuda_handle.cuh>
#include <mfmg/cuda/sparse_matrix_device.cuh>

#if MFMG_WITH_AMGX
#include <unordered_map>

#include <amgx_c.h>
#endif

namespace mfmg
{
template <typename VectorType>
class CudaSolver : public Solver<VectorType>
{
public:
  using value_type = typename VectorType::value_type;
  using vector_type = VectorType;

  CudaSolver(CudaHandle const &cuda_handle,
             std::shared_ptr<Operator<vector_type> const> op,
             std::shared_ptr<boost::property_tree::ptree const> params);

  ~CudaSolver();

  void apply(vector_type const &b, vector_type &x) const final;

private:
  CudaHandle const &_cuda_handle;
  std::string _solver;
#if MFMG_WITH_AMGX
  // AMGX handles and data
  AMGX_config_handle _amgx_config_handle;
  AMGX_resources_handle _amgx_res_handle;
  AMGX_matrix_handle _amgx_matrix_handle;
  AMGX_vector_handle _amgx_rhs_handle;
  AMGX_vector_handle _amgx_solution_handle;
  AMGX_solver_handle _amgx_solver_handle;
  int _device_id[1];
  std::unordered_map<int, int> _row_map;
#else
  void *_amgx_config_handle;
  void *_amgx_res_handle;
  void *_amgx_matrix_handle;
  void *_amgx_rhs_handle;
  void *_amgx_solution_handle;
  void *_amgx_solver_handle;
#endif
};
} // namespace mfmg

#endif

#endif
