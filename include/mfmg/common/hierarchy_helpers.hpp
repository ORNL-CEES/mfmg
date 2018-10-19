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

#ifndef MFMG_HIERARCHY_HELPERS_HPP
#define MFMG_HIERARCHY_HELPERS_HPP

#include <mfmg/common/mesh_evaluator.hpp>
#include <mfmg/common/operator.hpp>
#include <mfmg/common/smoother.hpp>
#include <mfmg/common/solver.hpp>

#include <boost/property_tree/ptree.hpp>

#include <memory>

#include <mpi.h>

namespace mfmg
{
template <typename VectorType>
class HierarchyHelpers
{
public:
  using vector_type = VectorType;

  virtual std::shared_ptr<Operator<vector_type>>
  get_global_operator(std::shared_ptr<MeshEvaluator> mesh_evaluator) = 0;

  virtual std::shared_ptr<Operator<vector_type>> build_restrictor(
      MPI_Comm comm, std::shared_ptr<MeshEvaluator> mesh_evaluator,
      std::shared_ptr<boost::property_tree::ptree const> params) = 0;

  virtual std::shared_ptr<Smoother<vector_type>>
  build_smoother(std::shared_ptr<Operator<vector_type> const> op,
                 std::shared_ptr<boost::property_tree::ptree const> params) = 0;

  virtual std::shared_ptr<Solver<vector_type>> build_coarse_solver(
      std::shared_ptr<Operator<vector_type> const> op,
      std::shared_ptr<boost::property_tree::ptree const> params) = 0;
};
} // namespace mfmg

#endif
