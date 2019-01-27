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

#include <mfmg/cuda/amge_device.templates.cuh>
#include <mfmg/cuda/cuda_mesh_evaluator.cuh>
#include <mfmg/cuda/vector_device.cuh>

#include <deal.II/lac/la_parallel_vector.h>

// Cannot use the instantiation macro with nvcc
template class mfmg::AMGe_device<
    2, mfmg::CudaMeshEvaluator<2>,
    dealii::LinearAlgebra::distributed::Vector<double>>;
template class mfmg::AMGe_device<2, mfmg::CudaMeshEvaluator<2>,
                                 mfmg::VectorDevice<double>>;
template class mfmg::AMGe_device<
    3, mfmg::CudaMeshEvaluator<3>,
    dealii::LinearAlgebra::distributed::Vector<double>>;
template class mfmg::AMGe_device<3, mfmg::CudaMeshEvaluator<3>,
                                 mfmg::VectorDevice<double>>;
