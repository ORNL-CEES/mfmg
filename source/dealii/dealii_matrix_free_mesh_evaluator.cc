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

#include <mfmg/common/instantiation.hpp>
#include <mfmg/dealii/dealii_matrix_free_mesh_evaluator.hpp>

namespace mfmg
{
template <int dim>
DealIIMatrixFreeMeshEvaluator<dim>::DealIIMatrixFreeMeshEvaluator(
    dealii::DoFHandler<dim> &dof_handler,
    dealii::AffineConstraints<double> &constraints)
    : DealIIMeshEvaluator<dim>(dof_handler, constraints,
                               "DealIIMatrixFreeMeshEvaluator")
{
}
} // namespace mfmg

// Explicit Instantiation
INSTANTIATE_DIM(TUPLE(DealIIMatrixFreeMeshEvaluator))
