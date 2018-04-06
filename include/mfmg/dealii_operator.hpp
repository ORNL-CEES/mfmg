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

#ifndef MFMG_DEALII_OPERATOR_HPP
#define MFMG_DEALII_OPERATOR_HPP

#include <mfmg/concepts.hpp>
#include <mfmg/exceptions.hpp>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <string>

namespace mfmg
{

template <typename VectorType>
class DealIIMatrixOperator : public MatrixOperator<VectorType>
{
public:
  using value_type = typename VectorType::value_type;
  using sparsity_pattern_type = dealii::SparsityPattern;
  using matrix_type = dealii::SparseMatrix<value_type>;
  using vector_type = VectorType;
  using operator_type = MatrixOperator<vector_type>;

  DealIIMatrixOperator(std::shared_ptr<matrix_type> matrix,
                       std::shared_ptr<sparsity_pattern_type> sparsity_pattern);

  virtual size_t m() const override final { return _matrix->m(); }

  virtual size_t n() const override final { return _matrix->n(); }

  size_t nnz() const { return _matrix->n_nonzero_elements(); }

  virtual void apply(vector_type const &x, vector_type &y) const override final;

  virtual std::shared_ptr<operator_type> transpose() const override final;

  virtual std::shared_ptr<operator_type>
  multiply(operator_type const &operator_b) const override final;

  std::shared_ptr<matrix_type> get_matrix() const { return _matrix; }

  virtual std::shared_ptr<vector_type>
  build_domain_vector() const override final;

  virtual std::shared_ptr<vector_type>
  build_range_vector() const override final;

private:
  // The sparsity pattern needs to outlive the sparse matrix, so we declare it
  // first. This is only necessary for dealii for deal.II sparse matrix.
  // Trilinos saves the Epetra_Map inside the sparse matrix.
  std::shared_ptr<sparsity_pattern_type> _sparsity_pattern;

  std::shared_ptr<matrix_type> _matrix;
};

template <typename VectorType>
class DealIITrilinosMatrixOperator : public MatrixOperator<VectorType>
{
public:
  using value_type = typename VectorType::value_type;
  using sparsity_pattern_type = dealii::TrilinosWrappers::SparsityPattern;
  using matrix_type = dealii::TrilinosWrappers::SparseMatrix;
  using vector_type = VectorType;
  using operator_type = MatrixOperator<vector_type>;

  DealIITrilinosMatrixOperator(
      std::shared_ptr<matrix_type> matrix,
      std::shared_ptr<sparsity_pattern_type> sparsity_pattern = nullptr);

  virtual size_t m() const override final { return _matrix->m(); }

  virtual size_t n() const override final { return _matrix->n(); }

  size_t nnz() const { return _matrix->n_nonzero_elements(); }

  virtual void apply(vector_type const &x, vector_type &y) const override final;

  virtual std::shared_ptr<operator_type> transpose() const override final;

  virtual std::shared_ptr<operator_type>
  multiply(operator_type const &operator_b) const override final;

  std::shared_ptr<matrix_type> get_matrix() const { return _matrix; }

  virtual std::shared_ptr<vector_type>
  build_domain_vector() const override final;

  std::shared_ptr<vector_type> build_range_vector() const override final;

private:
  std::shared_ptr<matrix_type> _matrix;
};

template <typename VectorType>
class DealIISmootherOperator : public Operator<VectorType>
{
public:
  using vector_type = VectorType;
  using matrix_type = dealii::TrilinosWrappers::SparseMatrix;
  using operator_type = Operator<vector_type>;

  DealIISmootherOperator(matrix_type const &matrix,
                         std::shared_ptr<boost::property_tree::ptree> params);

  virtual size_t m() const override final { return _matrix.m(); }

  virtual size_t n() const override final { return _matrix.n(); }

  virtual void apply(vector_type const &b, vector_type &x) const override final;

  virtual std::shared_ptr<VectorType>
  build_domain_vector() const override final;

  virtual std::shared_ptr<VectorType> build_range_vector() const override final;

private:
  void initialize(std::string const &prec_type);

  // Currently, dealii preconditioners do not support a mode to update a given
  // vector, instead replacing it. For smoothers, this mode is necessary to
  // operator. We workaround this issue by doing extra computations in "apply",
  // however this requires residual computation. Therefore, we need to store a
  // reference to matrix.
  matrix_type const &_matrix;
  std::unique_ptr<dealii::TrilinosWrappers::PreconditionBase> _smoother;
};

template <typename VectorType>
class DealIIDirectOperator : public Operator<VectorType>
{
public:
  using vector_type = VectorType;
  using matrix_type = dealii::TrilinosWrappers::SparseMatrix;
  using solver_type = dealii::TrilinosWrappers::SolverDirect;
  using operator_type = Operator<vector_type>;

  DealIIDirectOperator(matrix_type const &matrix);

  virtual size_t m() const override final { return _m; }

  virtual size_t n() const override final { return _n; }

  virtual void apply(vector_type const &b, vector_type &x) const override final
  {
    _solver->solve(x, b);
  }

  virtual std::shared_ptr<VectorType>
  build_domain_vector() const override final;

  virtual std::shared_ptr<VectorType> build_range_vector() const override final;

private:
  dealii::SolverControl _solver_control;
  std::unique_ptr<dealii::TrilinosWrappers::SolverDirect> _solver;
  size_t _m;
  size_t _n;
};
}

#endif
