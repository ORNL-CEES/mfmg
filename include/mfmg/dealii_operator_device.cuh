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

#ifndef MFMG_DEALII_OPERATOR_DEVICE_CUH
#define MFMG_DEALII_OPERATOR_DEVICE_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/concepts.hpp>
#include <mfmg/exceptions.hpp>
#include <mfmg/sparse_matrix_device.cuh>

#include <cusolverDn.h>
#include <cusolverSp.h>

namespace mfmg
{
template <typename VectorType>
class DirectDeviceOperator : public Operator<VectorType>
{
public:
  using value_type = typename VectorType::value_type;
  using vector_type = VectorType;
  using matrix_type = SparseMatrixDevice<value_type>;
  using operator_type = Operator<vector_type>;

  // Need to move the matrix to only one gpu -> need gather/scatter
  DirectDeviceOperator(cusolverDnHandle_t cusolver_dn_handle,
                       cusolverSpHandle_t cusolver_sp_handle,
                       matrix_type const &matrix, std::string const &solver);

  virtual size_t m() const override final { return _matrix.m(); }

  virtual size_t n() const override final { return _matrix.n(); }

  virtual void apply(vector_type const &b, vector_type &x) const override final;

  virtual std::shared_ptr<vector_type>
  build_domain_vector() const override final;

  virtual std::shared_ptr<vector_type>
  build_range_vector() const override final;

private:
  cusolverDnHandle_t _cusolver_dn_handle;
  cusolverSpHandle_t _cusolver_sp_handle;

  matrix_type const &_matrix;

  std::string _solver;
};
}

#endif

#endif
