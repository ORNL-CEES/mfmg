/*************************************************************************
 * Copyright (c) 2018 by the mfmg authors                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#ifndef VECTOR_DEVICE_TEMPLATES_CUH
#define VECTOR_DEVICE_TEMPLATES_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/vector_device.cuh>

namespace mfmg
{
namespace internal
{
template <typename ScalarType>
__global__ void add(int const size, ScalarType alpha,
                    ScalarType const *const x_val, ScalarType *val)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
    val[idx] += alpha * x_val[idx];
}
}

template <typename ScalarType>
VectorDevice<ScalarType>::VectorDevice(
    dealii::LinearAlgebra::distributed::Vector<ScalarType> const
        &distributed_vector)
{
  partitioner = distributed_vector.get_partitioner();
  unsigned int const size = partitioner->local_size();
  cuda_malloc(val_dev, size);

  // We need to cast away the const because of cudaMemcpy
  std::vector<ScalarType> val_host(distributed_vector.begin(),
                                   distributed_vector.end());
  cuda_mem_copy_to_dev(val_host, val_dev);
}

template <typename ScalarType>
VectorDevice<ScalarType>::VectorDevice(VectorDevice const &other)
{
  partitioner = other.partitioner;
  unsigned int const size = partitioner->local_size();
  cuda_malloc(val_dev, size);

  cudaError_t cuda_error_code;
  cuda_error_code =
      cudaMemcpy(val_dev, other.val_dev, size * sizeof(ScalarType),
                 cudaMemcpyDeviceToDevice);
  ASSERT_CUDA(cuda_error_code);
}

template <typename ScalarType>
VectorDevice<ScalarType>::~VectorDevice()
{
  if (val_dev == nullptr)
  {
    cuda_free(val_dev);
    val_dev = nullptr;
  }
}

template <typename ScalarType>
void VectorDevice<ScalarType>::add(ScalarType alpha,
                                   VectorDevice<ScalarType> const &x)
{
  int const local_size = partitioner->local_size();
  ASSERT(local_size == x.partitioner->local_size(),
         "The vectors don't have the same local size.");

  int n_blocks = 1 + (local_size - 1) / block_size;
  internal::add<<<n_blocks, block_size>>>(local_size, alpha, x.val_dev,
                                          val_dev);
}
}

#endif

#endif
