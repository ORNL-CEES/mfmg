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

#ifndef CUDA_HANDLE_CUH
#define CUDA_HANDLE_CUH

#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>

namespace mfmg
{
/**
 * This structure creates, stores, and destroys the handles of the different
 * CUDA libraries used inside deal.II.
 */
struct CudaHandle
{
  /**
   * Constructor. Create the handles for the different libraries.
   */
  CudaHandle();

  /**
   * Copy constructor is deleted.
   */
  CudaHandle(CudaHandle const &) = delete;

  /**
   * Destructor. Destroy the handles and free all the memory allocated by
   * GrowingVectorMemory.
   */
  ~CudaHandle();

  cusolverDnHandle_t cusolver_dn_handle;

  cusolverSpHandle_t cusolver_sp_handle;

  cusparseHandle_t cusparse_handle;
};
} // namespace mfmg

#endif
