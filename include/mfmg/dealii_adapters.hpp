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

#ifndef MFMG_DEALII_ADAPTERS_HPP
#define MFMG_DEALII_ADAPTERS_HPP

#include <mfmg/amge_host.hpp>
#include <mfmg/concepts.hpp>
#include <mfmg/dealii_mesh.hpp>
#include <mfmg/dealii_operator.hpp>
#include <mfmg/exceptions.hpp>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_solver.h>

#include <boost/property_tree/ptree.hpp>

#include <random>

#ifndef MFMG_USE_MATRIXFREE_GLOBAL_OPERATOR
#define MFMG_USE_MATRIXFREE_GLOBAL_OPERATOR 1
#endif

namespace mfmg
{

template <int dim, typename VectorType>
class DealIIMeshEvaluator
{
public:
  using mesh_type = DealIIMesh<dim>;
  using vector_type = VectorType;
  using value_type = typename VectorType::value_type;
  using operator_type = Operator<vector_type>;
#if MFMG_USE_MATRIXFREE_GLOBAL_OPERATOR
  using global_operator_type = DealIIMatrixFreeOperator<vector_type>;
#else
  using global_operator_type = DealIITrilinosMatrixOperator<vector_type>;
#endif
  using local_operator_type = DealIIMatrixOperator<dealii::Vector<value_type>>;

  DealIIMeshEvaluator() = default;

  // FIXME: should really be "const mesh_type& mesh, but right now `evaluate`
  // actually modifies the `dof_handler` due to `distribute_dofs
  std::shared_ptr<global_operator_type const>
  get_global_operator(mesh_type &mesh) const;

  std::shared_ptr<local_operator_type const>
  get_local_operator(mesh_type &mesh) const;

  // For deal.II, because of the way it deals with hanging nodes and
  // Dirichlet b.c., we need to zero out the initial guess values
  // corresponding to those. Otherwise, it may cause issues with spurious
  // modes and some scaling difficulties. If we use `apply` for deal.II, we
  // don't need this as we can apply the constraints immediately after
  // applying the matrix and before any norms and dot products.

  // In addition, this has to work only for ARPACK with dealii::Vector<double>
  void set_initial_guess(mesh_type const &mesh,
                         typename local_operator_type::vector_type &x) const;

protected:
  virtual void evaluate(dealii::DoFHandler<dim> &, dealii::ConstraintMatrix &,
                        dealii::TrilinosWrappers::SparsityPattern &,
                        dealii::TrilinosWrappers::SparseMatrix &) const = 0;

  virtual void evaluate(dealii::DoFHandler<dim> &, dealii::ConstraintMatrix &,
                        dealii::SparsityPattern &,
                        dealii::SparseMatrix<value_type> &) const = 0;
};

template <int dim, typename VectorType>
std::shared_ptr<
    typename DealIIMeshEvaluator<dim, VectorType>::global_operator_type const>
DealIIMeshEvaluator<dim, VectorType>::get_global_operator(mesh_type &mesh) const
{
  using matrix_type = typename global_operator_type::matrix_type;
  using sparsity_pattern_type =
      typename global_operator_type::sparsity_pattern_type;

  auto system_sparsity_pattern = std::make_shared<sparsity_pattern_type>();
  auto system_matrix = std::make_shared<matrix_type>();

  // Call user function to fill in the system matrix
  evaluate(mesh._dof_handler, mesh._constraints, *system_sparsity_pattern,
           *system_matrix);

  return std::make_shared<global_operator_type>(system_matrix,
                                                system_sparsity_pattern);
}

template <int dim, typename VectorType>
std::shared_ptr<
    typename DealIIMeshEvaluator<dim, VectorType>::local_operator_type const>
DealIIMeshEvaluator<dim, VectorType>::get_local_operator(mesh_type &mesh) const
{
  using matrix_type = typename local_operator_type::matrix_type;
  using sparsity_pattern_type =
      typename local_operator_type::sparsity_pattern_type;

  auto system_sparsity_pattern = std::make_shared<sparsity_pattern_type>();
  auto system_matrix = std::make_shared<matrix_type>();

  // Call user function to fill in the system matrix
  evaluate(mesh._dof_handler, mesh._constraints, *system_sparsity_pattern,
           *system_matrix);

  return std::make_shared<local_operator_type>(system_matrix,
                                               system_sparsity_pattern);
}

template <int dim, typename VectorType>
void DealIIMeshEvaluator<dim, VectorType>::set_initial_guess(
    typename DealIIMeshEvaluator<dim, VectorType>::mesh_type const &mesh,
    typename DealIIMeshEvaluator<
        dim, VectorType>::local_operator_type::vector_type &x) const
{
  unsigned int const n = x.size();

  std::default_random_engine generator;
  std::uniform_real_distribution<value_type> distribution(0., 1.);
  for (unsigned int i = 0; i < n; ++i)
    x[i] =
        (mesh._constraints.is_constrained(i) == false ? distribution(generator)
                                                      : 0.);
}

// This is a specialization of Adapter in concepts.hpp
template <int dim, typename VectorType>
class Adapter<DealIIMeshEvaluator<dim, VectorType>>
{
public:
  using mesh_evaluator_type = DealIIMeshEvaluator<dim, VectorType>;
  using mesh_type = typename mesh_evaluator_type::mesh_type;
  using operator_type = typename mesh_evaluator_type::operator_type;
  using global_operator_type =
      typename mesh_evaluator_type::global_operator_type;
  using smoother_type = DealIISmootherOperator<VectorType>;
  using direct_solver_type = DealIIDirectOperator<VectorType>;
  using vector_type = typename operator_type::vector_type;

  static std::shared_ptr<operator_type>
  build_restrictor(MPI_Comm comm, mesh_evaluator_type const &evaluator,
                   mesh_type &mesh,
                   std::shared_ptr<boost::property_tree::ptree> params)
  {
    auto eigensolver_params = params->get_child("eigensolver");
    AMGe_host<mesh_type::dimension(), mesh_evaluator_type, vector_type> amge(
        comm, mesh._dof_handler, eigensolver_params.get("type", "arpack"));

    auto agglomerate_params = params->get_child("agglomeration");
    int n_eigenvectors = eigensolver_params.get("number of eigenvectors", 1);
    double tolerance = eigensolver_params.get("tolerance", 1e-14);

    auto restrictor_matrix =
        std::make_shared<typename global_operator_type::matrix_type>();
    auto global_operator = evaluator.get_global_operator(mesh);
    amge.setup_restrictor(agglomerate_params, n_eigenvectors, tolerance,
                          evaluator, global_operator, *restrictor_matrix);

    return std::make_shared<global_operator_type>(restrictor_matrix);
  }

  static std::shared_ptr<operator_type>
  build_smoother(operator_type const &op,
                 std::shared_ptr<boost::property_tree::ptree> params)
  {
    auto global_op = dynamic_cast<global_operator_type const &>(op);
    return std::make_shared<smoother_type>(*global_op.get_matrix(), params);
  }

  static std::shared_ptr<operator_type>
  build_coarse_solver(operator_type const &op, mesh_evaluator_type const &,
                      std::shared_ptr<boost::property_tree::ptree> params)
  {
    auto global_op = dynamic_cast<global_operator_type const &>(op);
    return std::make_shared<direct_solver_type>(*global_op.get_matrix(),
                                                params);
  }
};

} // namespace mfmg

#endif // MFMG_ADAPTERS_DEALII
