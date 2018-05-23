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

#include <mfmg/cuda_handle.cuh>

#include <mfmg/exceptions.hpp>

namespace mfmg
{
CudaHandle::CudaHandle()
    : cusolver_dn_handle(nullptr), cusolver_sp_handle(nullptr),
      cusparse_handle(nullptr)
{
  // Create the dense cuSOLVER handle
  cusolverStatus_t cusolver_error_code = cusolverDnCreate(&cusolver_dn_handle);
  ASSERT_CUSOLVER(cusolver_error_code);
  // Create the sparse cuSOLVER handle
  cusolver_error_code = cusolverSpCreate(&cusolver_sp_handle);
  ASSERT_CUSOLVER(cusolver_error_code);
  // Create the cuSPARSE handle
  cusparseStatus_t cusparse_error_code = cusparseCreate(&cusparse_handle);
  ASSERT_CUSPARSE(cusparse_error_code);
}

CudaHandle::~CudaHandle()
{
  if (cusparse_handle != nullptr)
  {
    cusparseStatus_t cusparse_error_code = cusparseDestroy(cusparse_handle);
    ASSERT_CUSPARSE(cusparse_error_code);
    cusparse_handle = nullptr;
  }

  if (cusolver_sp_handle != nullptr)
  {
    cusolverStatus_t cusolver_error_code =
        cusolverSpDestroy(cusolver_sp_handle);
    ASSERT_CUSOLVER(cusolver_error_code);
    cusolver_sp_handle = nullptr;
  }

  if (cusolver_dn_handle != nullptr)
  {
    cusolverStatus_t cusolver_error_code =
        cusolverDnDestroy(cusolver_dn_handle);
    ASSERT_CUSOLVER(cusolver_error_code);
    cusolver_dn_handle = nullptr;
  }
}
}
