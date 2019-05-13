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

#ifndef MFMG_HIERARCHY_HPP
#define MFMG_HIERARCHY_HPP

#include <mfmg/common/exceptions.hpp>
#include <mfmg/common/level.hpp>
#include <mfmg/common/mesh_evaluator.hpp>
#include <mfmg/common/utils.hpp>
#include <mfmg/dealii/dealii_hierarchy_helpers.hpp>
#include <mfmg/dealii/dealii_matrix_free_hierarchy_helpers.hpp>
#ifdef MFMG_WITH_CUDA
#include <mfmg/cuda/cuda_hierarchy_helpers.cuh>
#include <mfmg/cuda/cuda_matrix_operator.cuh>
#include <mfmg/cuda/cuda_mesh_evaluator.cuh>
#endif

#include <deal.II/base/timer.h>

#include <boost/property_tree/ptree.hpp>

#include <memory>
#include <vector>

namespace mfmg
{
void timer_enter_subsection(std::shared_ptr<dealii::TimerOutput> const &timer,
                            std::string const &section)
{
  if (timer)
    timer->enter_subsection(section);
}

void timer_leave_subsection(std::shared_ptr<dealii::TimerOutput> const &timer)
{
  if (timer)
    timer->leave_subsection();
}

template <typename VectorType>
std::unique_ptr<HierarchyHelpers<VectorType>>
create_hierarchy_helpers(std::shared_ptr<MeshEvaluator const> evaluator)
{
  std::unique_ptr<HierarchyHelpers<VectorType>> hierarchy_helpers;
  std::string evaluator_type = evaluator->get_mesh_evaluator_type();
  if (evaluator_type == "DealIIMeshEvaluator")
  {
    int const dim = evaluator->get_dim();
    if (dim == 2)
      hierarchy_helpers.reset(new DealIIHierarchyHelpers<2, VectorType>());
    else if (dim == 3)
      hierarchy_helpers.reset(new DealIIHierarchyHelpers<3, VectorType>());
    else
      ASSERT_THROW_NOT_IMPLEMENTED();
  }
  else if (evaluator_type == "DealIIMatrixFreeMeshEvaluator")
  {
    int const dim = evaluator->get_dim();
    if (dim == 2)
      hierarchy_helpers.reset(
          new DealIIMatrixFreeHierarchyHelpers<2, VectorType>());
    else if (dim == 3)
      hierarchy_helpers.reset(
          new DealIIMatrixFreeHierarchyHelpers<3, VectorType>());
    else
      ASSERT_THROW_NOT_IMPLEMENTED();
  }
#ifdef MFMG_WITH_CUDA
  else if (evaluator_type == "CudaMeshEvaluator")
  {
    int const dim = evaluator->get_dim();

    if (dim == 2)
    {
      // Downcast evaluator
      auto const cuda_evaluator =
          std::dynamic_pointer_cast<CudaMeshEvaluator<2> const>(evaluator);
      hierarchy_helpers.reset(new CudaHierarchyHelpers<2, VectorType>(
          cuda_evaluator->get_cuda_handle()));
    }
    else if (dim == 3)
    {
      // Downcast evaluator
      auto const cuda_evaluator =
          std::dynamic_pointer_cast<CudaMeshEvaluator<3> const>(evaluator);
      hierarchy_helpers.reset(new CudaHierarchyHelpers<3, VectorType>(
          cuda_evaluator->get_cuda_handle()));
    }
    else
      ASSERT_THROW_NOT_IMPLEMENTED();
  }
#endif
  else
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }
  return hierarchy_helpers;
}

#ifdef MFMG_WITH_CUDA
template <>
std::unique_ptr<HierarchyHelpers<dealii::LinearAlgebra::distributed::Vector<
    double, dealii::MemorySpace::CUDA>>>
create_hierarchy_helpers(std::shared_ptr<MeshEvaluator const> evaluator)
{
  std::unique_ptr<HierarchyHelpers<dealii::LinearAlgebra::distributed::Vector<
      double, dealii::MemorySpace::CUDA>>>
      hierarchy_helpers;
  std::string evaluator_type = evaluator->get_mesh_evaluator_type();
  if (evaluator_type == "CudaMeshEvaluator")
  {
    int const dim = evaluator->get_dim();

    if (dim == 2)
    {
      // Downcast evaluator
      auto const cuda_evaluator =
          std::dynamic_pointer_cast<CudaMeshEvaluator<2> const>(evaluator);
      hierarchy_helpers.reset(
          new CudaHierarchyHelpers<2,
                                   dealii::LinearAlgebra::distributed::Vector<
                                       double, dealii::MemorySpace::CUDA>>(
              cuda_evaluator->get_cuda_handle()));
    }
    else if (dim == 3)
    {
      // Downcast evaluator
      auto const cuda_evaluator =
          std::dynamic_pointer_cast<CudaMeshEvaluator<3> const>(evaluator);
      hierarchy_helpers.reset(
          new CudaHierarchyHelpers<3,
                                   dealii::LinearAlgebra::distributed::Vector<
                                       double, dealii::MemorySpace::CUDA>>(
              cuda_evaluator->get_cuda_handle()));
    }
    else
      ASSERT_THROW_NOT_IMPLEMENTED();
  }
  else
    ASSERT_THROW_NOT_IMPLEMENTED();

  return hierarchy_helpers;
}
#endif

template <typename VectorType>
class Hierarchy
{
public:
  Hierarchy(MPI_Comm comm, std::shared_ptr<MeshEvaluator> evaluator,
            std::shared_ptr<boost::property_tree::ptree> params = nullptr,
            std::shared_ptr<dealii::TimerOutput> timer = nullptr)
      : _timer(timer)
  {
    timer_enter_subsection(_timer, "Setup");
    // Replace by a factory
    auto hierarchy_helpers = create_hierarchy_helpers<VectorType>(evaluator);

    _is_preconditioner = params->get("is preconditioner", true);
    _n_smoothing_steps = params->get("smoother.n_smoothing_steps", 1);

    // TODO: add stopping criteria for levels (number of levels / coarse size)
    const int num_levels = params->get("max levels", 2);
    _levels.resize(num_levels);

    _levels[0].set_operator(hierarchy_helpers->get_global_operator(evaluator));
    for (int level_index = 0; level_index < num_levels; level_index++)
    {
      auto &level_fine = _levels[level_index];

      auto a = level_fine.get_operator();

      if (level_index == num_levels - 1)
      {
        timer_enter_subsection(_timer, "Setup: build coarse solver");
        auto coarse_solver = hierarchy_helpers->build_coarse_solver(a, params);
        level_fine.set_solver(coarse_solver);
        timer_leave_subsection(_timer);

        break;
      }

      auto &level_coarse = _levels[level_index + 1];

      timer_enter_subsection(_timer, "Setup: build smoother");
      auto smoother = hierarchy_helpers->build_smoother(a, params);
      level_fine.set_smoother(smoother);
      timer_leave_subsection(_timer);

      timer_enter_subsection(_timer, "Setup: build restrictor");
      auto restrictor =
          hierarchy_helpers->build_restrictor(comm, evaluator, params);
      level_coarse.set_restrictor(restrictor);
      timer_leave_subsection(_timer);

      std::shared_ptr<Operator<VectorType>> ap;
      bool fast_ap = params->get("fast_ap", false);
      if (fast_ap)
      {
        timer_enter_subsection(_timer, "Setup: fast_ap");
        ap = hierarchy_helpers->fast_multiply_transpose();
        timer_leave_subsection(_timer);
      }
      else
      {
        timer_enter_subsection(_timer, "Setup: ap");
        ap = a->multiply_transpose(restrictor);
        timer_leave_subsection(_timer);
      }

      timer_enter_subsection(_timer, "Setup: build coarse matrix");
      auto a_coarse = restrictor->multiply(ap);
      timer_leave_subsection(_timer);

      level_coarse.set_operator(a_coarse);
    }
    timer_leave_subsection(_timer);
  }

  void vmult(VectorType &x, VectorType const &b) const
  {
    // Apply calls itself recursively which trips the timer so put it here
    timer_enter_subsection(_timer, "Apply");
    apply(b, x, 0);
    timer_leave_subsection(_timer);
  }

  void apply(VectorType const &b, VectorType &x, int level_index = 0) const
  {
    auto const num_levels = _levels.size();

    auto &level_fine = _levels[level_index];
    auto a = level_fine.get_operator();

    if (level_index > 0 || _is_preconditioner)
    {
      // Zero out any garbage in x.
      // The only exception is when it's the finest level in a standalone
      // mode.
      x = 0.;
    }

    if (level_index == num_levels - 1)
    {
      timer_enter_subsection(_timer, "Apply: coarsest level");
      // Coarsest level
      auto coarse_solver = level_fine.get_solver();
      coarse_solver->apply(b, x);
      timer_leave_subsection(_timer);
    }
    else
    {
      timer_enter_subsection(_timer, "Apply: fine levels");
      auto &level_coarse = _levels[level_index + 1];

      auto restrictor = level_coarse.get_restrictor();

      // apply pre-smoother
      auto smoother = level_fine.get_smoother();
      for (unsigned int i = 0; i < _n_smoothing_steps; ++i)
        smoother->apply(b, x);

      // compute residual
      // NOTE: we compute negative residual -r = Ax-b, so that we can avoid
      // using sadd and can just use add
      auto res = level_fine.build_vector();
      a->apply(x, *res);
      res->add(-1., b);

      // restrict residual
      auto b_coarse = level_coarse.build_vector();
      restrictor->apply(*res, *b_coarse);

      // compute coarse grid correction
      auto x_coarse = level_coarse.build_vector();
      apply(*b_coarse, *x_coarse, level_index + 1);

      // update solution
      auto x_correction = level_fine.build_vector();
      restrictor->apply(*x_coarse, *x_correction, OperatorMode::TRANS);

      // NOTE: as we used negative residual, we subtract instead of adding
      // here
      x.add(-1., *x_correction);

      // apply post-smoother
      for (unsigned int i = 0; i < _n_smoothing_steps; ++i)
        smoother->apply(b, x);
      timer_leave_subsection(_timer);
    }
  }

  double grid_complexity() const
  {
    // auto const num_levels = _levels.size();

    // if (num_levels == 0)
    //   return -1.0;

    // auto level0_m = _levels[0].get_operator()->grid_complexity();
    // ASSERT(level0_m, "The size of the finest level operator is 0.");

    // double complexity = level0_m;

    // for (int i = 1; i < num_levels; i++)
    // {
    //   if (i < num_levels - 1)
    //     complexity += _levels[i].get_operator()->grid_complexity();
    //   else
    //   {
    //     // Hierarchy may be continued using a different multigrid
    //     // For direct solvers, this would be equivalent to using
    //     //   _levels[i].get_operator()->grid_complexity()
    //     complexity += _levels[i].get_smoother()->grid_complexity();
    //   }
    // }

    // return complexity / level0_m;

    return 0;
  }

  double operator_complexity() const
  {
    // auto const num_levels = _levels.size();

    // if (num_levels == 0)
    //   return -1.0;

    // auto level0_nnz = _levels[0].get_operator()->operator_complexity();
    // ASSERT(level0_nnz, "The nnz of the finest level operator is 0.");

    // double complexity = level0_nnz;
    // for (int i = 1; i < num_levels; i++)
    // {
    //   if (i < num_levels - 1)
    //     complexity += _levels[i].get_operator()->operator_complexity();
    //   else
    //   {
    //     // Hierarchy may be continued using a different multigrid
    //     // For direct solvers, this would be equivalent to using
    //     //   _levels[i].get_operator()->operator_complexity()
    //     complexity += _levels[i].get_smoother()->operator_complexity();
    //   }
    // }
    // return complexity / level0_nnz;
    return 0;
  }

private:
  std::shared_ptr<dealii::TimerOutput> _timer;
  std::vector<Level<VectorType>> _levels;
  bool _is_preconditioner = true;
  unsigned int _n_smoothing_steps;
};
} // namespace mfmg

#endif // ifdef MFMG_HIERARCHY_HPP
