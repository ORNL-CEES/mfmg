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

#define BOOST_TEST_MODULE laplace_matrix_free

#include "laplace_matrix_free.hpp"
#include "main.cc"

#include <deal.II/lac/precondition.h>

namespace tt = boost::test_tools;

template <int dim>
class ExactSolution : public dealii::Function<dim>
{
public:
  ExactSolution() = default;

  double value(dealii::Point<dim> const &p,
               unsigned int const component = 0) const override;
};

template <int dim>
double ExactSolution<dim>::value(dealii::Point<dim> const &p,
                                 unsigned int const) const
{
  double val = 1.;
  for (unsigned int d = 0; d < dim; ++d)
    val *= (p[d] - 1.) * p[d];

  return val;
}

template <int dim>
class Source
{
public:
  Source() = default;

  template <typename ScalarType>
  dealii::VectorizedArray<ScalarType>
  value(dealii::Point<dim, dealii::VectorizedArray<ScalarType>> const &p) const;
};

template <int dim>
template <typename ScalarType>
dealii::VectorizedArray<ScalarType> Source<dim>::value(
    dealii::Point<dim, dealii::VectorizedArray<ScalarType>> const &p) const
{
  auto const zero = dealii::make_vectorized_array<ScalarType>(0.);
  auto const one = dealii::make_vectorized_array<ScalarType>(1.);
  auto const two = dealii::make_vectorized_array<ScalarType>(2.);

  dealii::VectorizedArray<ScalarType> val = zero;
  for (unsigned int d = 0; d < dim; ++d)
  {
    dealii::VectorizedArray<ScalarType> tmp = zero;
    for (unsigned int i = 0; i < dim; ++i)
      if (i != d)
        tmp += p[i] * (p[i] - one);

    val -= two * tmp;
  }

  return val;
}

template <int dim>
class MaterialProperty
{
public:
  MaterialProperty() = default;

  template <typename ScalarType>
  dealii::VectorizedArray<ScalarType>
  value(dealii::Point<dim, dealii::VectorizedArray<ScalarType>> const &p) const;
};

template <int dim>
template <typename ScalarType>
dealii::VectorizedArray<ScalarType> MaterialProperty<dim>::value(
    dealii::Point<dim, dealii::VectorizedArray<ScalarType>> const &) const
{
  return dealii::make_vectorized_array<ScalarType>(1.);
}

BOOST_AUTO_TEST_CASE(laplace_2d)
{
  int constexpr dim = 2;
  int constexpr fe_degree = 2;

  MaterialProperty<dim> material_property;
  Source<dim> source;

  LaplaceMatrixFree<dim, fe_degree, double> laplace(MPI_COMM_WORLD);
  laplace.setup_system(boost::property_tree::ptree(), material_property);
  laplace.assemble_rhs(source);

  dealii::PreconditionIdentity preconditioner;
  laplace.solve(preconditioner);

  // The exact solution is quadratic so the error should be zero.
  ExactSolution<dim> exact_solution;
  BOOST_TEST(laplace.compute_error(exact_solution), tt::tolerance(1e-14));
}

BOOST_AUTO_TEST_CASE(laplace_3d)
{
  int constexpr dim = 3;
  int constexpr fe_degree = 2;

  MaterialProperty<dim> material_property;
  Source<dim> source;

  LaplaceMatrixFree<dim, fe_degree, double> laplace(MPI_COMM_WORLD);
  laplace.setup_system(boost::property_tree::ptree(), material_property);
  laplace.assemble_rhs(source);

  dealii::PreconditionIdentity preconditioner;
  laplace.solve(preconditioner);

  // The exact solution is quadratic so the error should be zero.
  ExactSolution<dim> exact_solution;
  BOOST_TEST(laplace.compute_error(exact_solution), tt::tolerance(1e-14));
}
