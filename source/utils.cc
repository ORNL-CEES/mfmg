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

#include <mfmg/exceptions.hpp>
#include <mfmg/utils.hpp>

#include <EpetraExt_MultiVectorOut.h>
#include <EpetraExt_RowMatrixOut.h>

#include <string>

namespace mfmg
{

// TODO: write down 4 maps
void matrix_market_output_file(
    const std::string &filename,
    const dealii::TrilinosWrappers::SparseMatrix &matrix)
{
  int rv = EpetraExt::RowMatrixToMatrixMarketFile(filename.c_str(),
                                                  matrix.trilinos_matrix());
  ASSERT(rv != 0, "EpetraExt::RowMatrixToMatrixMarketFile return value is " +
                      std::to_string(rv));
}

// TODO: write the map
void matrix_market_output_file(
    const std::string &filename,
    const dealii::TrilinosWrappers::MPI::Vector &vector)
{
  int rv = EpetraExt::MultiVectorToMatrixMarketFile(filename.c_str(),
                                                    vector.trilinos_vector());
  ASSERT(rv != 0, "EpetraExt::RowMatrixToMatrixMarketFile return value is " +
                      std::to_string(rv));
}
}
