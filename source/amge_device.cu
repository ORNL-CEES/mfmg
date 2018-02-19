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

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_solver.h>

// Cannot use the instantiation macro with nvcc
template class mfmg::AMGe_device<2, dealii::TrilinosWrappers::MPI::Vector>;
template class mfmg::AMGe_device<
    2, dealii::LinearAlgebra::distributed::Vector<float>>;
template class mfmg::AMGe_device<
    2, dealii::LinearAlgebra::distributed::Vector<double>>;
template class mfmg::AMGe_device<3, dealii::TrilinosWrappers::MPI::Vector>;
template class mfmg::AMGe_device<
    3, dealii::LinearAlgebra::distributed::Vector<float>>;
template class mfmg::AMGe_device<
    3, dealii::LinearAlgebra::distributed::Vector<double>>;
