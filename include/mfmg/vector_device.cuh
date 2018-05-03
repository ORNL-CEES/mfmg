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

#ifndef VECTOR_DEVICE_CUH
#define VECTOR_DEVICE_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/exceptions.hpp>
#include <mfmg/utils.cuh>

#include <deal.II/base/partitioner.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <memory>

namespace mfmg
{
/**
 * This structure encapsulates the pointer that defines the vector on the
 * device and the partitioner.
 */
template <typename ScalarType>
struct VectorDevice
{
  typedef ScalarType value_type;

  VectorDevice(std::shared_ptr<const dealii::Utilities::MPI::Partitioner> part)
      : partitioner(part)
  {
    cuda_malloc(val_dev, partitioner->local_size());
  }

  VectorDevice(dealii::LinearAlgebra::distributed::Vector<ScalarType> const
                   &distributed_vector);

  /**
   * Copy constructor. Note that the vectors share the same partitioner so
   * changing the partitioner of one vector will change the partitioner in the
   * other.
   */
  VectorDevice(VectorDevice const &other);

  ~VectorDevice();

  void add(ScalarType alpha, VectorDevice<ScalarType> const &x);

  ScalarType const *get_values() const { return val_dev; }

  ScalarType *get_values() { return val_dev; }

  ScalarType *val_dev;
  std::shared_ptr<const dealii::Utilities::MPI::Partitioner> partitioner;
};
}

#endif

#endif
