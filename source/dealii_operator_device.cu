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

#include <mfmg/dealii_operator_device.templates.cuh>
#include <mfmg/vector_device.cuh>

#include <deal.II/lac/la_parallel_vector.h>

template class mfmg::SparseMatrixDeviceOperator<mfmg::VectorDevice<double>>;
template class mfmg::SmootherDeviceOperator<mfmg::VectorDevice<double>>;
template class mfmg::DirectDeviceOperator<mfmg::VectorDevice<double>>;

template class mfmg::SparseMatrixDeviceOperator<
    dealii::LinearAlgebra::distributed::Vector<double>>;
template class mfmg::SmootherDeviceOperator<
    dealii::LinearAlgebra::distributed::Vector<double>>;
template class mfmg::DirectDeviceOperator<
    dealii::LinearAlgebra::distributed::Vector<double>>;
