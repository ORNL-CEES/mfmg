/*************************************************************************
 * Copyright (c) 2017-2018 by the mfmg authors                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * This file is part of the mfmg libary. mfmg is distributed under op BSD *
 * 3-clause license. For the licensing terms see the LICENSE file in the *
 * top-level directory                                                   *
 *                                                                       *
 * SPDX-License-Identifier: BSD-3-Clause                                 *
 *************************************************************************/

#ifndef MFMG_HIERARCHY_HPP
#define MFMG_HIERARCHY_HPP

#include <memory>
#include <vector>

#include <boost/property_tree/ptree.hpp>

#include "mfmg/utils.hpp"

namespace mfmg
{

template <typename OperatorType>
class Level
{
public:
  using operator_type = OperatorType;

public:
  std::shared_ptr<operator_type const> get_operator() const
  {
    return _operator;
  }

  std::shared_ptr<operator_type const> get_restrictor() const
  {
    return _restrictor;
  }

  std::shared_ptr<operator_type const> get_prolongator() const
  {
    return _prolongator;
  }

  std::shared_ptr<operator_type const> get_smoother() const
  {
    return _smoother;
  }

  void set_operator(std::shared_ptr<operator_type const> op) { _operator = op; }

  void set_restrictor(std::shared_ptr<operator_type const> r)
  {
    _restrictor = r;
  }

  void set_prolongator(std::shared_ptr<operator_type const> p)
  {
    _prolongator = p;
  }

  void set_smoother(std::shared_ptr<operator_type const> s) { _smoother = s; }

private:
  std::shared_ptr<operator_type const> _operator, _prolongator, _restrictor,
      _smoother;
};

template <typename MeshEvaluatorType, typename VectorType>
class Hierarchy
{
public:
  using mesh_evaluator_type = MeshEvaluatorType;
  using mesh_type = typename MeshEvaluatorType::mesh_type;
  using vector_type = VectorType;
  using operator_type = typename MeshEvaluatorType::operator_type;
  using global_operator_type = typename MeshEvaluatorType::global_operator_type;

public:
  Hierarchy(MPI_Comm comm, mesh_evaluator_type &evaluator, mesh_type &mesh,
            std::shared_ptr<boost::property_tree::ptree> params = nullptr)
  {
    static_assert(std::is_same<typename MeshEvaluatorType::vector_type,
                               vector_type>::value,
                  "Vector template does not match the one in MeshEvaluator");

    using HierarchyHelpers = Adapter<mesh_evaluator_type>;

    _is_preconditioner = params->get("is preconditioner", true);

    // TODO: add stopping criteria for levels (number of levels / coarse size)
    const int num_levels = params->get("max levels", 2);
    _levels.resize(num_levels);

    _levels[0].set_operator(evaluator.get_global_operator(mesh));
    for (int level_index = 0; level_index < num_levels; level_index++)
    {
      auto &level_fine = _levels[level_index];

      auto a = std::dynamic_pointer_cast<global_operator_type const>(
          level_fine.get_operator());

      if (level_index == num_levels - 1)
      {
        auto direct_solver = HierarchyHelpers::build_direct_solver(*a);
        level_fine.set_smoother(direct_solver);

        break;
      }

      auto &level_coarse = _levels[level_index + 1];

      auto smoother = HierarchyHelpers::build_smoother(*a, params);
      level_fine.set_smoother(smoother);

      auto restrictor =
          HierarchyHelpers::build_restrictor(comm, evaluator, mesh, params);
      level_coarse.set_restrictor(restrictor);

      auto prolongator =
          std::dynamic_pointer_cast<global_operator_type>(restrictor)
              ->transpose();
      level_coarse.set_prolongator(prolongator);

      auto ap =
          std::dynamic_pointer_cast<global_operator_type const>(a)->multiply(
              *prolongator);
      auto a_coarse =
          std::dynamic_pointer_cast<global_operator_type>(restrictor)
              ->multiply(*ap);
      level_coarse.set_operator(a_coarse);
    }
  }

  // TODO: should this go to some kind of deal.ii mfmg adapter? This is deal.ii
  // specific.
  void vmult(vector_type &x, vector_type const &b) const { apply(b, x, 0); }

  void apply(vector_type const &b, vector_type &x, int level_index = 0) const
  {
    auto const num_levels = _levels.size();

    auto &level_fine = _levels[level_index];
    auto a = level_fine.get_operator();

    if (level_index > 0 || _is_preconditioner)
    {
      // Zero out any garbage in x.
      // The only exception is when it's the finest level in a standalone mode.
      x = 0.;
    }

    if (level_index == num_levels - 1)
    {
      // Coarsest level

      // Direct solver = smoother
      auto smoother = level_fine.get_smoother();
      smoother->apply(b, x);
    }
    else
    {
      auto &level_coarse = _levels[level_index + 1];

      auto prolongator = level_coarse.get_prolongator();
      auto restrictor = level_coarse.get_restrictor();

      // apply pre-smoother
      auto smoother = level_fine.get_smoother();
      smoother->apply(b, x);

      // compute residual
      // NOTE: we compute negative residual -r = Ax-b, so that we can avoid
      // using sadd and can just use add
      auto res = a->build_range_vector();
      a->apply(x, *res);
      res->add(-1., b);

      // restrict residual
      auto b_coarse = restrictor->build_range_vector();
      restrictor->apply(*res, *b_coarse);

      // compute coarse grid correction
      auto x_coarse = prolongator->build_domain_vector();
      apply(*b_coarse, *x_coarse, level_index + 1);

      // update solution
      auto x_correction = prolongator->build_range_vector();
      prolongator->apply(*x_coarse, *x_correction);

      // NOTE: as we used negative residual, we subtract instead of adding here
      x.add(-1., *x_correction);

      // apply post-smoother
      smoother->apply(b, x);
    }
  }

  double grid_complexity() const
  {
    auto const num_levels = _levels.size();

    if (num_levels == 0)
      return -1.0;

    auto level0_m = _levels[0].get_operator()->m();
    ASSERT(level0_m, "The size of the finest level operator is 0.");

    double complexity = level0_m;
    for (int i = 1; i < num_levels; i++)
      complexity += _levels[i].get_operator()->m();

    return complexity / level0_m;
  }

  double operator_complexity() const
  {
    auto const num_levels = _levels.size();

    if (num_levels == 0)
      return -1.0;

    auto level0_nnz = std::dynamic_pointer_cast<global_operator_type const>(
                          _levels[0].get_operator())
                          ->nnz();
    ASSERT(level0_nnz, "The nnz of the finest level operator is 0.");

    double complexity = level0_nnz;
    for (int i = 1; i < num_levels; i++)
      complexity += std::dynamic_pointer_cast<const global_operator_type>(
                        _levels[i].get_operator())
                        ->nnz();
    return complexity / level0_nnz;
  }

private:
  std::vector<Level<operator_type>> _levels;
  bool _is_preconditioner = true;
};
}

#endif // ifdef MFMG_HIERARCHY_HPP
