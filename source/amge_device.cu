/*************************************************************************
 * Copyright (c) 2017 by the mfmg authors                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#include <mfmg/amge_device.templates.cuh>

// Cannot use the instantiation macro with nvcc
template class mfmg::AMGe_device<2, float>;
template class mfmg::AMGe_device<2, double>;
template class mfmg::AMGe_device<3, float>;
template class mfmg::AMGe_device<3, double>;
