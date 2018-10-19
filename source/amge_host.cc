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

#include <mfmg/amge_host.templates.hpp>

template class mfmg::AMGe_host<2, mfmg::DealIIMeshEvaluator<2>,
                               dealii::TrilinosWrappers::MPI::Vector>;
template class mfmg::AMGe_host<
    2, mfmg::DealIIMeshEvaluator<2>,
    dealii::LinearAlgebra::distributed::Vector<double>>;
template class mfmg::AMGe_host<3, mfmg::DealIIMeshEvaluator<3>,
                               dealii::TrilinosWrappers::MPI::Vector>;
template class mfmg::AMGe_host<
    3, mfmg::DealIIMeshEvaluator<3>,
    dealii::LinearAlgebra::distributed::Vector<double>>;
