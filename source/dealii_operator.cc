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

#include <mfmg/dealii_operator.templates.hpp>
#include <mfmg/instantiation.hpp>

INSTANTIATE_SERIALVECTORTYPE(TUPLE(dealii_adapter::DealIIMatrixOperator))
INSTANTIATE_VECTORTYPE(TUPLE(dealii_adapter::TrilinosMatrixOperator))
INSTANTIATE_VECTORTYPE(TUPLE(dealii_adapter::SmootherOperator))
INSTANTIATE_VECTORTYPE(TUPLE(dealii_adapter::DirectOperator))
