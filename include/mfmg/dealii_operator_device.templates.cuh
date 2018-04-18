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

#ifndef MFMG_DEALII_OPERATOR_DEVICE_TEMPLATES_CUH
#define MFMG_DEALII_OPERATOR_DEVICE_TEMPLATES_CUH

#include <mfmg/dealii_operator_device.cuh>
#include <mfmg/dealii_operator_device_helpers.cuh>
#include <mfmg/utils.cuh>

#include <EpetraExt_Transpose_RowMatrix.h>

#include <algorithm>

namespace mfmg
{
template <typename VectorType>
DirectDeviceOperator<VectorType>::DirectDeviceOperator(
    cusolverDnHandle_t cusolver_dn_handle,
    cusolverSpHandle_t cusolver_sp_handle,
    SparseMatrixDevice<typename VectorType::value_type> const &matrix,
    std::string const &solver)
    : _cusolver_dn_handle(cusolver_dn_handle),
      _cusolver_sp_handle(cusolver_sp_handle), _matrix(matrix), _solver(solver)
{
  // Transform to lower case
  std::transform(_solver.begin(), _solver.end(), _solver.begin(), tolower);
}

template <typename VectorType>
void DirectDeviceOperator<VectorType>::apply(VectorType const &b,
                                             VectorType &x) const
{
  if (_solver == "cholesky")
    cholesky_factorization(_cusolver_sp_handle, _matrix, b.get_values(),
                           x.get_values());
  else if (_solver == "lu_dense")
    lu_factorization(_cusolver_dn_handle, _matrix, b.get_values(),
                     x.get_values());
  else if (_solver == "lu_sparse_host")
    lu_factorization(_cusolver_sp_handle, _matrix, b.get_values(),
                     x.get_values());
  else
    ASSERT_THROW(false, "The provided solver name " + _solver + " is invalid.");
}

template <typename VectorType>
std::shared_ptr<VectorType>
DirectDeviceOperator<VectorType>::build_domain_vector() const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
}

template <typename VectorType>
std::shared_ptr<VectorType>
DirectDeviceOperator<VectorType>::build_range_vector() const
{
  ASSERT_THROW_NOT_IMPLEMENTED();

  return nullptr;
}
}

#endif
