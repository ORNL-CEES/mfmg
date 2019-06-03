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

#ifndef MFMG_CUDA_MATRIX_FREE_MESH_EVALUATOR_CUH
#define MFMG_CUDA_MATRIX_FREE_MESH_EVALUATOR_CUH

#ifdef MFMG_WITH_CUDA

#include <mfmg/cuda/cuda_mesh_evaluator.cuh>

namespace mfmg
{
template <int dim>
class CudaMatrixFreeMeshEvaluator : public CudaMeshEvaluator<dim>
{
public:
  using size_type = unsigned int;

  static int constexpr _dim = dim;

  CudaMatrixFreeMeshEvaluator(CudaHandle const &cuda_handle,
                              dealii::DoFHandler<dim> &dof_handler,
                              dealii::AffineConstraints<double> &constraints)
      : CudaMeshEvaluator<dim>(cuda_handle, dof_handler, constraints)
  {
  }

  virtual std::string get_mesh_evaluator_type() const override final;

  void apply(dealii::LinearAlgebra::distributed::Vector<
                 double, dealii::MemorySpace::CUDA> const &src,
             dealii::LinearAlgebra::distributed::Vector<
                 double, dealii::MemorySpace::CUDA> &dst) const;

  void apply(dealii::LinearAlgebra::distributed::Vector<
                 double, dealii::MemorySpace::Host> const &src,
             dealii::LinearAlgebra::distributed::Vector<
                 double, dealii::MemorySpace::Host> &dst) const;

  std::shared_ptr<dealii::LinearAlgebra::distributed::Vector<double>>
  build_range_vector() const;

  void set_initial_guess(dealii::AffineConstraints<double> & /*constraints*/,
                         dealii::LinearAlgebra::distributed::Vector<
                             double, dealii::MemorySpace::CUDA> & /*x*/) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  virtual void matrix_free_initialize_agglomerate(dealii::DoFHandler<dim> &
                                                  /*dof_handler*/) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  virtual void matrix_free_evaluate_agglomerate(
      dealii::LinearAlgebra::distributed::Vector<
          double, dealii::MemorySpace::CUDA> const & /*src*/,
      dealii::LinearAlgebra::distributed::Vector<
          double, dealii::MemorySpace::CUDA> & /*dst*/) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  virtual std::vector<double> matrix_free_get_agglomerate_diagonal(
      dealii::AffineConstraints<double> & /*constraints*/) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return std::vector<double>();
  }

  // TODO
  template <typename VectorType>
  VectorType get_diagonal_inverse() const
  {
    return VectorType();
  }
};

template <int dim>
struct is_matrix_free<CudaMatrixFreeMeshEvaluator<dim>> : std::true_type
{
};

} // namespace mfmg

#endif

#endif
