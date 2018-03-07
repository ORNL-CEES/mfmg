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

#ifndef MFMG_CONCEPTS_HPP
#define MFMG_CONCEPTS_HPP

#include <mfmg/exceptions.hpp>

#include <memory>

namespace mfmg
{

template <class VectorType>
class Operator
{
public:
  using operator_type = Operator<VectorType>;
  using vector_type = VectorType;

public:
  void apply(const vector_type &x, vector_type &y) const;
  std::shared_ptr<operator_type> transpose() const;
  std::shared_ptr<operator_type> multiply(const operator_type &b) const;

  std::shared_ptr<vector_type> build_domain_vector() const;
  std::shared_ptr<vector_type> build_range_vector() const;
};

template <class Mesh, class GlobalOperatorType,
          class LocalOperatorType = GlobalOperatorType>
class MeshEvaluator
{
public:
  using global_operator_type = GlobalOperatorType;
  using local_operator_type = LocalOperatorType;
  using mesh_type = Mesh;

public:
  std::shared_ptr<global_operator_type>
  get_global_operator(const mesh_type &mesh) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }
  std::shared_ptr<local_operator_type>
  get_local_operator(const mesh_type &mesh) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }
};

template <class MeshEvaluatorType>
class Adapter
{
public:
  using mesh_type = typename MeshEvaluatorType::mesh_type;
  using global_operator_type = typename MeshEvaluatorType::global_operator_type;
  using mesh_evaluator_type = MeshEvaluatorType;

public:
  std::shared_ptr<global_operator_type> build_restrictor(
      MPI_Comm comm, const mesh_evaluator_type &evaluator,
      const mesh_type &mesh,
      std::shared_ptr<const boost::property_tree::ptree> params) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }
};
}

#endif
