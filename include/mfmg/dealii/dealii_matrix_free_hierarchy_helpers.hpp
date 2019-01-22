/*************************************************************************
 * Copyright (c) 2017-2019 by the mfmg authors                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under a BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#ifndef MFMG_DEALII_MATRIX_FREE_HIERARCHY_HELPERS_HPP
#define MFMG_DEALII_MATRIX_FREE_HIERARCHY_HELPERS_HPP

#include <mfmg/dealii/dealii_hierarchy_helpers.hpp>

namespace mfmg
{
template <int dim, typename VectorType>
class DealIIMatrixFreeHierarchyHelpers
    : public DealIIHierarchyHelpers<dim, VectorType>
{
public:
  using vector_type = VectorType;

  std::shared_ptr<Operator<vector_type>> get_global_operator(
      std::shared_ptr<MeshEvaluator> mesh_evaluator) override final;

  std::shared_ptr<Operator<vector_type>> build_restrictor(
      MPI_Comm comm, std::shared_ptr<MeshEvaluator> mesh_evaluator,
      std::shared_ptr<boost::property_tree::ptree const> params) override final;

  std::shared_ptr<Smoother<vector_type>> build_smoother(
      std::shared_ptr<Operator<vector_type> const> op,
      std::shared_ptr<boost::property_tree::ptree const> params) override final;
};
} // namespace mfmg

#endif
