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

#define BOOST_TEST_MODULE laplace_matrix_free_device

#include <deal.II/base/cuda.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_precondition.h>

#include "laplace.hpp"
#include "laplace_matrix_free_device.cuh"
#include "main.cc"

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
  ScalarType value(dealii::Point<dim, ScalarType> const &p) const;
};

template <int dim>
template <typename ScalarType>
ScalarType Source<dim>::value(dealii::Point<dim, ScalarType> const &p) const
{
  ScalarType val = 0.;
  for (unsigned int d = 0; d < dim; ++d)
  {
    ScalarType tmp = 1.;
    for (unsigned int i = 0; i < dim; ++i)
      if (i != d)
        tmp *= (p[i] - 1.) * p[i];

    val -= 2. * tmp;
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

template <int dim>
class MSource : public dealii::Function<dim>
{
public:
  MSource() = default;

  double value(dealii::Point<dim> const &p,
               unsigned int const component = 0) const override;
};

template <int dim>
double MSource<dim>::value(dealii::Point<dim> const &p,
                           unsigned int const) const
{
  double val = 0.;
  for (unsigned int d = 0; d < dim; ++d)
  {
    double tmp = 1.;
    for (unsigned int i = 0; i < dim; ++i)
      if (i != d)
        tmp *= (p[i] - 1.) * p[i];

    val += -2. * tmp;
  }

  return val;
}

template <typename VectorType>
class MyCG : public dealii::SolverCG<VectorType>
{
public:
  MyCG(dealii::SolverControl &cn) : dealii::SolverCG<VectorType>(cn) {}

  void print_vectors(const unsigned int step, const VectorType &x,
                     const VectorType &r, const VectorType &d) const override
  {
    //  std::cout << "step " << step << std::endl;
    //  x.print(std::cout);
    //  r.print(std::cout);
    //  d.print(std::cout);
  }
};

BOOST_AUTO_TEST_CASE(laplace_2d)
{
  int constexpr dim = 2;
  int constexpr fe_degree = 3;

  dealii::Utilities::CUDA::Handle cuda_handle;
  Source<dim> source;

  LaplaceMatrixFreeDevice<dim, fe_degree, double> laplace_dev(MPI_COMM_WORLD);
  auto material_property =
      DeviceMaterialPropertyFactory<dim>::create_material_property("constant");
  auto material_property_host =
      MaterialPropertyFactory<dim>::create_material_property("constant");
  laplace_dev.setup_system(boost::property_tree::ptree(), *material_property,
                           *material_property_host);
  laplace_dev.assemble_rhs(source);

  dealii::PreconditionIdentity preconditioner;
  laplace_dev.solve(preconditioner);

  // The exact solution is quadratic so the error should be zero.
  ExactSolution<dim> exact_solution;
  BOOST_TEST(laplace_dev.compute_error(exact_solution) == 0.,
             tt::tolerance(1e-14));
}

BOOST_AUTO_TEST_CASE(laplace_3d)
{
  int constexpr dim = 3;
  int constexpr fe_degree = 2;

  dealii::Utilities::CUDA::Handle cuda_handle;
  Source<dim> source;

  LaplaceMatrixFreeDevice<dim, fe_degree, double> laplace_dev(MPI_COMM_WORLD);
  auto material_property =
      DeviceMaterialPropertyFactory<dim>::create_material_property("constant");
  auto material_property_host =
      MaterialPropertyFactory<dim>::create_material_property("constant");
  laplace_dev.setup_system(boost::property_tree::ptree(), *material_property,
                           *material_property_host);
  laplace_dev.assemble_rhs(source);
  std::cout << "rhs " << laplace_dev._system_rhs.l2_norm() << std::endl;
  auto solution = laplace_dev._system_rhs;

  dealii::PreconditionIdentity preconditioner;
  laplace_dev.solve(preconditioner);

  // The exact solution is quadratic so the error should be zero.
  ExactSolution<dim> exact_solution;
  BOOST_TEST(laplace_dev.compute_error(exact_solution) == 0.,
             tt::tolerance(1e-14));
}
