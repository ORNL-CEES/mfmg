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

#ifndef MFMG_ADAPTERS_DEALII_HPP
#define MFMG_ADAPTERS_DEALII_HPP

#include <boost/property_tree/ptree.hpp>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_solver.h>

#include "amge_host.hpp"

#include <mfmg/concepts.hpp>
#include <mfmg/dealii_operator.hpp>
#include <mfmg/exceptions.hpp>

#include <random>

namespace mfmg
{

template <int dim>
struct DealIIMesh
{
  DealIIMesh(dealii::DoFHandler<dim> &dof_handler,
             dealii::ConstraintMatrix &constraints)
      : _dof_handler(dof_handler), _constraints(constraints)
  {
  }

  static constexpr int dimension() { return dim; }
  dealii::DoFHandler<dim> &_dof_handler;
  dealii::ConstraintMatrix &_constraints;
};

template <int dim, class VectorType>
class DealIIMeshEvaluator
{
public:
  using mesh_type = DealIIMesh<dim>;
  using vector_type = VectorType;
  using value_type = typename VectorType::value_type;
  using operator_type = DealIIOperator<vector_type>;
  using global_operator_type =
      DealIIMatrixOperator<dealii::TrilinosWrappers::SparsityPattern,
                           dealii::TrilinosWrappers::SparseMatrix, vector_type>;
  using local_operator_type =
      DealIIMatrixOperator<dealii::SparsityPattern,
                           dealii::SparseMatrix<value_type>,
                           dealii::Vector<value_type>>;

protected:
  virtual void
  evaluate(dealii::DoFHandler<dim> &dof_handler,
           dealii::ConstraintMatrix &constraints,
           dealii::TrilinosWrappers::SparsityPattern &system_sparsity_pattern,
           dealii::TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }
  virtual void evaluate(dealii::DoFHandler<dim> &dof_handler,
                        dealii::ConstraintMatrix &constraints,
                        dealii::SparsityPattern &system_sparsity_pattern,
                        dealii::SparseMatrix<value_type> &system_matrix) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

public:
  DealIIMeshEvaluator() {}

  // FIXME: should really be "const mesh_type& mesh, but right now `evaluate`
  // actually modifies the `dof_handler` due to `distribute_dofs
  std::shared_ptr<const global_operator_type>
  get_global_operator(mesh_type &mesh) const
  {
    using matrix_type = typename global_operator_type::matrix_type;
    using sparsity_pattern_type =
        typename global_operator_type::sparsity_pattern_type;

    auto system_sparsity_pattern = std::make_shared<sparsity_pattern_type>();
    auto system_matrix = std::make_shared<matrix_type>();

    // Call user function to fill in the matrix and build the mass matrix
    evaluate(mesh._dof_handler, mesh._constraints, *system_sparsity_pattern,
             *system_matrix);

    return std::make_shared<global_operator_type>(system_matrix,
                                                  system_sparsity_pattern);
  }

  std::shared_ptr<const local_operator_type>
  get_local_operator(mesh_type &mesh) const
  {
    using matrix_type = typename local_operator_type::matrix_type;
    using sparsity_pattern_type =
        typename local_operator_type::sparsity_pattern_type;

    auto system_sparsity_pattern = std::make_shared<sparsity_pattern_type>();
    auto system_matrix = std::make_shared<matrix_type>();

    // Call user function to fill in the matrix and build the mass matrix
    evaluate(mesh._dof_handler, mesh._constraints, *system_sparsity_pattern,
             *system_matrix);

    return std::make_shared<local_operator_type>(system_matrix,
                                                 system_sparsity_pattern);
  }

  // For deal.II, because of the way it deals with hanging nodes and
  // Dirichlet b.c., we need to zero out the initial guess values
  // corresponding to those. Otherwise, it may cause issues with spurious
  // modes and some scaling difficulties. If we use `apply` for deal.II, we
  // don't need this as we can apply the constraints immediately after
  // applying the matrix and before any norms and dot products.
  //
  // In addition, this has to work only for ARPACK with dealii::Vector<double>
  void set_initial_guess(const mesh_type &mesh,
                         typename local_operator_type::vector_type &x) const
  {
    unsigned int const n = x.size();

    std::default_random_engine generator;
    std::uniform_real_distribution<value_type> distribution(0., 1.);
    for (unsigned int i = 0; i < n; ++i)
      x[i] = (mesh._constraints.is_constrained(i) == false
                  ? distribution(generator)
                  : 0.);
  }
};

template <int dim, class VectorType>
class Adapter<DealIIMeshEvaluator<dim, VectorType>>
{
protected:
  using mesh_evaluator_type = DealIIMeshEvaluator<dim, VectorType>;
  using mesh_type = typename mesh_evaluator_type::mesh_type;
  using operator_type = typename mesh_evaluator_type::operator_type;
  using global_operator_type =
      typename mesh_evaluator_type::global_operator_type;
  using smoother_type = DealIISmootherOperator<VectorType>;
  using direct_solver_type = DealIIDirectOperator<VectorType>;
  using vector_type = typename operator_type::vector_type;

public:
  static std::shared_ptr<operator_type>
  build_restrictor(MPI_Comm comm, const mesh_evaluator_type &evaluator,
                   const mesh_type &mesh,
                   std::shared_ptr<boost::property_tree::ptree> params)
  {
    AMGe_host<mesh_type::dimension(), mesh_evaluator_type, vector_type> amge(
        comm, mesh._dof_handler);

    std::array<unsigned int, dim> agglomerate_dim;
    agglomerate_dim[0] = params->get<unsigned int>("agglomeration: nx");
    agglomerate_dim[1] = params->get<unsigned int>("agglomeration: ny");
    if (dim == 3)
      agglomerate_dim[2] = params->get<unsigned int>("agglomeration: nz");
    int n_eigenvectors =
        params->get<int>("eigensolver: number of eigenvectors", 1);
    double tolerance = params->get<double>("eigensolver: tolerance", 1e-14);

    auto restrictor_matrix =
        std::make_shared<typename global_operator_type::matrix_type>();
    amge.setup_restrictor(agglomerate_dim, n_eigenvectors, tolerance, evaluator,
                          *restrictor_matrix);

    return std::make_shared<global_operator_type>(restrictor_matrix);
  }

  static std::shared_ptr<operator_type>
  build_smoother(const operator_type &op,
                 std::shared_ptr<boost::property_tree::ptree> params)
  {
    auto global_op = dynamic_cast<const global_operator_type &>(op);
    return std::make_shared<smoother_type>(*global_op.get_matrix(), params);
  }

  static std::shared_ptr<operator_type>
  build_direct_solver(const operator_type &op)
  {
    auto global_op = dynamic_cast<const global_operator_type &>(op);
    return std::make_shared<direct_solver_type>(*global_op.get_matrix());
  }
};

} // namespace mfmg

#endif // MFMG_ADAPTERS_DEALII
