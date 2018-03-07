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

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <EpetraExt_Transpose_RowMatrix.h>

#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <string>

namespace mfmg
{

template <class VectorType>
class DealIIOperator
{
public:
  using vector_type = VectorType;

public:
  virtual void apply(const vector_type &x, vector_type &y) const = 0;

  virtual std::shared_ptr<vector_type> build_domain_vector() const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return nullptr;
  }

  virtual std::shared_ptr<vector_type> build_range_vector() const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return nullptr;
  }
};

template <class SparsityPatternType, class MatrixType, class VectorType>
class DealIIMatrixOperator : public DealIIOperator<VectorType>
{
public:
  using sparsity_pattern_type = SparsityPatternType;
  using matrix_type = MatrixType;
  using vector_type = VectorType;
  using operator_type =
      DealIIMatrixOperator<sparsity_pattern_type, matrix_type, vector_type>;

public:
  DealIIMatrixOperator(
      std::shared_ptr<matrix_type> matrix,
      std::shared_ptr<sparsity_pattern_type> sparsity_pattern = nullptr)
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  virtual void apply(const vector_type &x, vector_type &y) const override final
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  std::shared_ptr<operator_type> transpose() const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  std::shared_ptr<operator_type> multiply(const operator_type &operator_b) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  std::shared_ptr<matrix_type> get_matrix() const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  virtual std::shared_ptr<vector_type>
  build_domain_vector() const override final
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }

  virtual std::shared_ptr<vector_type> build_range_vector() const override final
  {
    ASSERT_THROW_NOT_IMPLEMENTED();
  }
};

template <class VectorType>
class DealIIMatrixOperator<
    dealii::SparsityPattern,
    dealii::SparseMatrix<typename VectorType::value_type>, VectorType>
    : public DealIIOperator<VectorType>
{
public:
  using value_type = typename VectorType::value_type;
  using sparsity_pattern_type = dealii::SparsityPattern;
  using matrix_type = dealii::SparseMatrix<value_type>;
  using vector_type = VectorType;
  using operator_type =
      DealIIMatrixOperator<sparsity_pattern_type, matrix_type, vector_type>;

public:
  DealIIMatrixOperator(
      std::shared_ptr<matrix_type> matrix,
      std::shared_ptr<sparsity_pattern_type> sparsity_pattern = nullptr)
      : _sparsity_pattern(sparsity_pattern), _matrix(matrix)
  {
  }

  virtual void apply(const vector_type &x, vector_type &y) const override final
  {
    _matrix->vmult(y, x);
  }

  std::shared_ptr<operator_type> transpose() const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return nullptr;
  }

  std::shared_ptr<operator_type> multiply(const operator_type &operator_b) const
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return nullptr;
  }

  std::shared_ptr<matrix_type> get_matrix() const { return _matrix; }

  virtual std::shared_ptr<vector_type>
  build_domain_vector() const override final
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return nullptr;
  }

  virtual std::shared_ptr<vector_type> build_range_vector() const override final
  {
    ASSERT_THROW_NOT_IMPLEMENTED();

    return nullptr;
  }

private:
  // TODO: for some reason, order here is important. If _sparsity_pattern after
  // _matrix, it results in some dealii subscriptor throws.
  std::shared_ptr<sparsity_pattern_type>
      _sparsity_pattern; // this may only be necessary for dealii::SparseMatrix,
                         // which does not store the pattern. For Trilinos
                         // matrices, this is not needed.
  std::shared_ptr<matrix_type> _matrix;
};

template <class VectorType>
class DealIIMatrixOperator<dealii::TrilinosWrappers::SparsityPattern,
                           dealii::TrilinosWrappers::SparseMatrix, VectorType>
    : public DealIIOperator<VectorType>
{
public:
  using value_type = typename VectorType::value_type;
  using sparsity_pattern_type = dealii::TrilinosWrappers::SparsityPattern;
  using matrix_type = dealii::TrilinosWrappers::SparseMatrix;
  using vector_type = VectorType;
  using operator_type =
      DealIIMatrixOperator<sparsity_pattern_type, matrix_type, vector_type>;

public:
  DealIIMatrixOperator(
      std::shared_ptr<matrix_type> matrix,
      std::shared_ptr<sparsity_pattern_type> sparsity_pattern = nullptr)
      : _sparsity_pattern(sparsity_pattern), _matrix(matrix)
  {
  }

  virtual void apply(const vector_type &x, vector_type &y) const override final
  {
    _matrix->vmult(y, x);
  }

  std::shared_ptr<operator_type> transpose() const
  {
    auto epetra_matrix = _matrix->trilinos_matrix();

    EpetraExt::RowMatrix_Transpose transposer;
    auto transposed_epetra_matrix =
        dynamic_cast<Epetra_CrsMatrix &>(transposer(epetra_matrix));

    auto transposed_matrix = std::make_shared<matrix_type>();
    transposed_matrix->reinit(transposed_epetra_matrix);

    return std::make_shared<operator_type>(transposed_matrix);
  }

  std::shared_ptr<operator_type> multiply(const operator_type &operator_b) const
  {
    auto a = this->get_matrix();
    auto b = operator_b.get_matrix();

    auto c = std::make_shared<matrix_type>();
    a->mmult(*c, *b);

    return std::make_shared<operator_type>(c);
  }

  std::shared_ptr<matrix_type> get_matrix() const { return _matrix; }

  virtual std::shared_ptr<vector_type>
  build_domain_vector() const override final
  {
    return std::make_shared<vector_type>(
        _matrix->locally_owned_domain_indices(),
        _matrix->get_mpi_communicator());
  }

  std::shared_ptr<vector_type> build_range_vector() const override final
  {
    return std::make_shared<vector_type>(_matrix->locally_owned_range_indices(),
                                         _matrix->get_mpi_communicator());
  }

private:
  // TODO: for some reason, order here is important. If _sparsity_pattern after
  // _matrix, it results in some dealii subscriptor throws.
  std::shared_ptr<sparsity_pattern_type>
      _sparsity_pattern; // this may only be necessary for dealii::SparseMatrix,
                         // which does not store the pattern. For Trilinos
                         // matrices, this is not needed.
  std::shared_ptr<matrix_type> _matrix;
};

template <class VectorType>
class DealIISmootherOperator : public DealIIOperator<VectorType>
{
public:
  using vector_type = VectorType;
  using matrix_type = dealii::TrilinosWrappers::SparseMatrix;

private:
  enum SmootherType
  {
    SGS,
    GS,
    JACOBI,
    ILU
  };

public:
  DealIISmootherOperator(const matrix_type &matrix,
                         std::shared_ptr<boost::property_tree::ptree> params)
      : _matrix(matrix)
  {
    std::string prec_name = params->get<std::string>("preconditioner: type",
                                                     "Symmetric Gauss-Seidel");
    _prec_type = string2type(prec_name);
    initialize(_prec_type);
  }

  void apply(const vector_type &b, vector_type &x) const
  {
    // r = -(b - Ax)
    vector_type r(b);
    _matrix.vmult(r, x);
    r.add(-1., b);

    // x = x + B^{-1} (-r)
    vector_type tmp(x);
    smoother_vmult(tmp, r);
    x.add(-1., tmp);
  }

private:
  SmootherType string2type(const std::string &prec_name) const
  {
    // Make parameters case-insensitive
    std::string prec_name_lower = prec_name;
    std::transform(prec_name_lower.begin(), prec_name_lower.end(),
                   prec_name_lower.begin(), ::tolower);
    if (prec_name_lower == "symmetric gauss-seidel")
      return SGS;
    else if (prec_name_lower == "gauss-seidel")
      return GS;
    else if (prec_name_lower == "jacobi")
      return JACOBI;
    else if (prec_name_lower == "ilu")
      return ILU;
    else
      throw std::runtime_error("Unknown smoother name: \"" + prec_name_lower +
                               "\"");
  }
  void initialize(const SmootherType prec_type)
  {
    switch (prec_type)
    {
    case SGS:
      _sgs_smoother.initialize(_matrix);
      break;
    case GS:
      _gs_smoother.initialize(_matrix);
      break;
    case JACOBI:
      _jacobi_smoother.initialize(_matrix);
      break;
    case ILU:
      _ilu_smoother.initialize(_matrix);
      break;
    };
  }

  void smoother_vmult(vector_type &x, const vector_type &b) const
  {
    {
      switch (_prec_type)
      {
      case SGS:
        _sgs_smoother.vmult(x, b);
        break;
      case GS:
        _gs_smoother.vmult(x, b);
        break;
      case JACOBI:
        _jacobi_smoother.vmult(x, b);
        break;
      case ILU:
        _ilu_smoother.vmult(x, b);
        break;
      };
    }
  }

private:
  const matrix_type &_matrix;
  SmootherType _prec_type;
  dealii::TrilinosWrappers::PreconditionSSOR _sgs_smoother;
  dealii::TrilinosWrappers::PreconditionSOR _gs_smoother;
  dealii::TrilinosWrappers::PreconditionJacobi _jacobi_smoother;
  dealii::TrilinosWrappers::PreconditionILU _ilu_smoother;
};

template <class VectorType>
class DealIIDirectOperator : public DealIIOperator<VectorType>
{
public:
  using vector_type = VectorType;
  using matrix_type = dealii::TrilinosWrappers::SparseMatrix;
  using solver_type = dealii::TrilinosWrappers::SolverDirect;

public:
  DealIIDirectOperator(const matrix_type &a)
  {
    _solver = std::make_shared<solver_type>(_solver_control);
    _solver->initialize(a);
  }

  void apply(const vector_type &b, vector_type &x) const
  {
    _solver->solve(x, b);
  }

private:
  dealii::SolverControl _solver_control;
  std::shared_ptr<dealii::TrilinosWrappers::SolverDirect> _solver;
};
}

#endif
