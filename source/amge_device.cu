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

#include <mfmg/amge_device.templates.cuh>

#include <mfmg/dealii_adapters_device.cuh>

#include <deal.II/lac/la_parallel_vector.h>

// Cannot use the instantiation macro with nvcc
template class mfmg::AMGe_device<
    2,
    mfmg::DealIIMeshEvaluatorDevice<
        2, dealii::LinearAlgebra::distributed::Vector<double>>,
    dealii::LinearAlgebra::distributed::Vector<double>>;
template class mfmg::AMGe_device<
    3,
    mfmg::DealIIMeshEvaluatorDevice<
        3, dealii::LinearAlgebra::distributed::Vector<double>>,
    dealii::LinearAlgebra::distributed::Vector<double>>;
