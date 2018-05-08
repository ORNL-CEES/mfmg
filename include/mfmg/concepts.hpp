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

#include <boost/property_tree/ptree.hpp>

#include <memory>
#include <mpi.h>

namespace mfmg
{

template <typename VectorType>
class Operator
{
public:
  using operator_type = Operator<VectorType>;
  using vector_type = VectorType;

  virtual size_t m() const = 0;
  virtual size_t n() const = 0;

  virtual size_t grid_complexity() const = 0;
  virtual size_t operator_complexity() const = 0;

  virtual void apply(vector_type const &x, vector_type &y) const = 0;

  virtual std::shared_ptr<vector_type> build_domain_vector() const = 0;
  virtual std::shared_ptr<vector_type> build_range_vector() const = 0;
};

template <typename VectorType>
class MatrixOperator : public Operator<VectorType>
{
public:
  using operator_type = MatrixOperator<VectorType>;
  using vector_type = VectorType;

  virtual std::shared_ptr<operator_type> transpose() const = 0;
  virtual std::shared_ptr<operator_type>
  multiply(operator_type const &b) const = 0;
};

template <typename Mesh, typename GlobalOperatorType,
          typename LocalOperatorType = GlobalOperatorType>
class MeshEvaluator
{
public:
  using global_operator_type = GlobalOperatorType;
  using local_operator_type = LocalOperatorType;
  using mesh_type = Mesh;

  std::shared_ptr<global_operator_type>
  get_global_operator(mesh_type const &) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return nullptr;
  }

  std::shared_ptr<local_operator_type>
  get_local_operator(mesh_type const &) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return nullptr;
  }
};

template <typename MeshEvaluatorType>
class Adapter
{
public:
  using mesh_type = typename MeshEvaluatorType::mesh_type;
  using operator_type = typename MeshEvaluatorType::operator_type;
  using global_operator_type = typename MeshEvaluatorType::global_operator_type;
  using mesh_evaluator_type = MeshEvaluatorType;

  static std::shared_ptr<operator_type>
  build_restrictor(MPI_Comm, mesh_evaluator_type const &, mesh_type const &,
                   std::shared_ptr<boost::property_tree::ptree const>)
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return nullptr;
  }

  static std::shared_ptr<operator_type>
  build_smoother(operator_type const &,
                 std::shared_ptr<boost::property_tree::ptree>)
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return nullptr;
  }

  static std::shared_ptr<operator_type>
  build_coarse_solver(operator_type const &, mesh_evaluator_type const &,
                      std::shared_ptr<boost::property_tree::ptree>)
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return nullptr;
  }
};
}

#endif
