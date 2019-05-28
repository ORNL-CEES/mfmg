/**************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                            *
 * All rights reserved.                                                   *
 *                                                                        *
 * This file is part of the mfmg library. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the  *
 * top-level directory                                                    *
 *                                                                        *
 * SPDX-License-Identifier: BSD-3-Clause                                  *
 *************************************************************************/

#ifndef MFMG_MESH_EVALUATOR_HPP
#define MFMG_MESH_EVALUATOR_HPP

#include <string>

namespace mfmg
{
class MeshEvaluator
{
public:
  virtual ~MeshEvaluator() = default;

  virtual int get_dim() const = 0;

  virtual std::string get_mesh_evaluator_type() const = 0;
};
} // namespace mfmg

#endif
