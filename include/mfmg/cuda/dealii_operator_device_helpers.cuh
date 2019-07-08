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

#ifndef MFMG_DEALII_OPERATOR_DEVICE_HELPERS_CUH
#define MFMG_DEALII_OPERATOR_DEVICE_HELPERS_CUH

#include <mfmg/cuda/sparse_matrix_device.cuh>

#include <cusolverSp.h>

/**
 * These functions are helper functions. There are mainly wrappers around
 * cuSOLVER functions.
 */

namespace mfmg
{
void cholesky_factorization(cusolverSpHandle_t cusolver_sp_handle,
                            SparseMatrixDevice<float> const &matrix,
                            float const *b, float *x);

void cholesky_factorization(cusolverSpHandle_t cusolver_sp_handle,
                            SparseMatrixDevice<double> const &matrix,
                            double const *b, double *x);
template <typename ScalarType>
void lu_factorization(cusolverDnHandle_t cusolver_dn_handle,
                      SparseMatrixDevice<ScalarType> const &matrix,
                      ScalarType const *b, ScalarType *x);

template <typename ScalarType>
void lu_factorization(cusolverSpHandle_t cusolver_sp_handle,
                      SparseMatrixDevice<ScalarType> const &matrix,
                      ScalarType const *b, ScalarType *x);

__global__ void iota(int const size, int *data, int const value = 0);
} // namespace mfmg

#endif
